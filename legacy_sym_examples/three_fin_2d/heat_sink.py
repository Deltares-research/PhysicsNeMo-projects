# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

import hydra
import numpy as np
import torch
from omegaconf import DictConfig as PhysicsNeMoConfig
from torch import nn
from torch.optim import Adam, lr_scheduler


class FlowHeatNet(nn.Module):
    def __init__(self, hidden_dim: int, num_layers: int):
        super().__init__()
        layers = [nn.Linear(2, hidden_dim), nn.Tanh()]
        for _ in range(max(0, num_layers - 1)):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
        self.body = nn.Sequential(*layers)
        self.flow_head = nn.Linear(hidden_dim, 3)
        self.temp_head = nn.Linear(hidden_dim, 1)

    def forward(self, xy: torch.Tensor):
        h = self.body(xy)
        flow = self.flow_head(h)
        temp = self.temp_head(h)
        return flow[:, 0:1], flow[:, 1:2], flow[:, 2:3], temp


def grad(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y), create_graph=True)[0]


def ns_residual(model: FlowHeatNet, xy: torch.Tensor, nu: float):
    xy = xy.requires_grad_(True)
    u, v, p, _ = model(xy)
    u_x = grad(u, xy)[:, 0:1]
    u_y = grad(u, xy)[:, 1:2]
    v_x = grad(v, xy)[:, 0:1]
    v_y = grad(v, xy)[:, 1:2]
    p_x = grad(p, xy)[:, 0:1]
    p_y = grad(p, xy)[:, 1:2]
    u_xx = grad(u_x, xy)[:, 0:1]
    u_yy = grad(u_y, xy)[:, 1:2]
    v_xx = grad(v_x, xy)[:, 0:1]
    v_yy = grad(v_y, xy)[:, 1:2]
    continuity = u_x + v_y
    mx = u * u_x + v * u_y + p_x - nu * (u_xx + u_yy)
    my = u * v_x + v * v_y + p_y - nu * (v_xx + v_yy)
    return continuity, mx, my


def heat_residual(model: FlowHeatNet, xy: torch.Tensor, d: float):
    xy = xy.requires_grad_(True)
    u, v, _, c = model(xy)
    c_x = grad(c, xy)[:, 0:1]
    c_y = grad(c, xy)[:, 1:2]
    c_xx = grad(c_x, xy)[:, 0:1]
    c_yy = grad(c_y, xy)[:, 1:2]
    return u * c_x + v * c_y - d * (c_xx + c_yy)


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def run(cfg: PhysicsNeMoConfig) -> None:
    torch.manual_seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x_bounds = (-2.5, 2.5)
    y_bounds = (-0.5, 0.5)

    model = FlowHeatNet(int(cfg.model.hidden_dim), int(cfg.model.num_layers)).to(device)
    optimizer = Adam(model.parameters(), lr=float(cfg.optimizer.lr))
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=float(cfg.scheduler.decay_rate))
    mse = nn.MSELoss()

    nu = float(cfg.physics.nu)
    diffusivity = float(cfg.physics.diffusivity)
    inlet_vel = float(cfg.physics.inlet_velocity)
    heat_sink_temp = float(cfg.physics.heat_sink_temp)

    def sample_rect(n: int, xb=x_bounds, yb=y_bounds):
        x = torch.empty(n, 1, device=device).uniform_(xb[0], xb[1])
        y = torch.empty(n, 1, device=device).uniform_(yb[0], yb[1])
        return torch.cat([x, y], dim=1)

    for step in range(int(cfg.training.max_steps)):
        optimizer.zero_grad()

        interior_flow = sample_rect(int(cfg.batch_size.interior_flow))
        c, mx, my = ns_residual(model, interior_flow, nu)
        z = torch.zeros_like(c)
        loss_flow = mse(c, z) + mse(mx, z) + mse(my, z)

        interior_heat = sample_rect(int(cfg.batch_size.interior_heat))
        rh = heat_residual(model, interior_heat, diffusivity)
        loss_heat = mse(rh, torch.zeros_like(rh))

        inlet = sample_rect(int(cfg.batch_size.inlet), xb=(x_bounds[0], x_bounds[0]))
        inlet[:, 0:1] = x_bounds[0]
        u_i, v_i, _, c_i = model(inlet)
        loss_in = mse(u_i, torch.full_like(u_i, inlet_vel)) + mse(v_i, torch.zeros_like(v_i)) + mse(c_i, torch.zeros_like(c_i))

        outlet = sample_rect(int(cfg.batch_size.outlet), xb=(x_bounds[1], x_bounds[1]))
        outlet[:, 0:1] = x_bounds[1]
        _, _, p_o, _ = model(outlet)
        loss_out = mse(p_o, torch.zeros_like(p_o))

        hs_wall = sample_rect(int(cfg.batch_size.hs_wall), xb=(-1.0, 0.0), yb=(-0.3, 0.3))
        u_h, v_h, _, c_h = model(hs_wall)
        loss_hs = mse(u_h, torch.zeros_like(u_h)) + mse(v_h, torch.zeros_like(v_h)) + mse(c_h, torch.full_like(c_h, heat_sink_temp))

        ch_wall = sample_rect(int(cfg.batch_size.channel_wall))
        top = torch.rand(ch_wall.shape[0], device=device) > 0.5
        ch_wall[top, 1] = y_bounds[1]
        ch_wall[~top, 1] = y_bounds[0]
        u_w, v_w, _, _ = model(ch_wall)
        loss_wall = mse(u_w, torch.zeros_like(u_w)) + mse(v_w, torch.zeros_like(v_w))

        loss = (
            float(cfg.loss.inlet) * loss_in
            + float(cfg.loss.outlet) * loss_out
            + float(cfg.loss.hs_wall) * loss_hs
            + float(cfg.loss.channel_wall) * loss_wall
            + float(cfg.loss.interior_flow) * loss_flow
            + float(cfg.loss.interior_heat) * loss_heat
        )

        loss.backward()
        optimizer.step()
        scheduler.step()

        if step % int(cfg.training.log_every) == 0:
            print(f"step={step:06d} total={loss.item():.4e} inlet={loss_in.item():.4e} outlet={loss_out.item():.4e}")


if __name__ == "__main__":
    run()
