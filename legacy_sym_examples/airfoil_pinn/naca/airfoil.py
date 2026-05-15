# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

import hydra
import numpy as np
import torch
from omegaconf import DictConfig as PhysicsNeMoConfig
from torch import nn
from torch.optim import Adam, lr_scheduler


class AirfoilFlowNet(nn.Module):
    def __init__(self, hidden_dim: int, num_layers: int):
        super().__init__()
        layers = [nn.Linear(5, hidden_dim), nn.Tanh()]
        for _ in range(max(0, num_layers - 1)):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
        layers.append(nn.Linear(hidden_dim, 3))
        self.net = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor):
        out = self.net(inputs)
        return out[:, 0:1], out[:, 1:2], out[:, 2:3]


def grad(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y), create_graph=True)[0]


def ns_residual(model: AirfoilFlowNet, inp: torch.Tensor, nu: float):
    inp = inp.requires_grad_(True)
    u, v, p = model(inp)
    x = inp[:, 0:1]
    y = inp[:, 1:2]
    xy = torch.cat([x, y], dim=1)

    u_x = grad(u, inp)[:, 0:1]
    u_y = grad(u, inp)[:, 1:2]
    v_x = grad(v, inp)[:, 0:1]
    v_y = grad(v, inp)[:, 1:2]
    p_x = grad(p, inp)[:, 0:1]
    p_y = grad(p, inp)[:, 1:2]

    u_xx = grad(u_x, inp)[:, 0:1]
    u_yy = grad(u_y, inp)[:, 1:2]
    v_xx = grad(v_x, inp)[:, 0:1]
    v_yy = grad(v_y, inp)[:, 1:2]

    continuity = u_x + v_y
    mx = u * u_x + v * u_y + p_x - nu * (u_xx + u_yy)
    my = u * v_x + v * v_y + p_y - nu * (v_xx + v_yy)
    return continuity, mx, my


def sample_param_batch(n: int, device):
    alpha = torch.empty(n, 1, device=device).uniform_(-0.2, 0.0)
    camber = torch.empty(n, 1, device=device).uniform_(0.0, 0.2)
    thickness = torch.empty(n, 1, device=device).uniform_(0.1, 0.2)
    return alpha, camber, thickness


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def run(cfg: PhysicsNeMoConfig) -> None:
    torch.manual_seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x_bounds = (-2.25, 5.25)
    y_bounds = (-2.5, 2.5)
    nu = float(cfg.physics.nu)
    u_inlet = float(cfg.physics.u_inlet)

    model = AirfoilFlowNet(int(cfg.model.hidden_dim), int(cfg.model.num_layers)).to(device)
    optimizer = Adam(model.parameters(), lr=float(cfg.optimizer.lr))
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=float(cfg.scheduler.decay_rate))
    mse = nn.MSELoss()

    def make_inputs(n: int, x_fixed=None):
        x = torch.empty(n, 1, device=device).uniform_(x_bounds[0], x_bounds[1])
        if x_fixed is not None:
            x[:] = x_fixed
        y = torch.empty(n, 1, device=device).uniform_(y_bounds[0], y_bounds[1])
        alpha, camber, thickness = sample_param_batch(n, device)
        return torch.cat([x, y, alpha, camber, thickness], dim=1)

    for step in range(int(cfg.training.max_steps)):
        optimizer.zero_grad()

        interior_far = make_inputs(int(cfg.batch_size.interior_far_field))
        c1, mx1, my1 = ns_residual(model, interior_far, nu)
        z1 = torch.zeros_like(c1)
        loss_far = mse(c1, z1) + mse(mx1, z1) + mse(my1, z1)

        interior_near = make_inputs(int(cfg.batch_size.interior_near_field))
        c2, mx2, my2 = ns_residual(model, interior_near, nu)
        z2 = torch.zeros_like(c2)
        loss_near = mse(c2, z2) + mse(mx2, z2) + mse(my2, z2)

        inlet = make_inputs(int(cfg.batch_size.inlet), x_fixed=x_bounds[0])
        u_i, v_i, _ = model(inlet)
        loss_in = mse(u_i, torch.full_like(u_i, u_inlet)) + mse(v_i, torch.zeros_like(v_i))

        outlet = make_inputs(int(cfg.batch_size.outlet), x_fixed=x_bounds[1])
        _, _, p_o = model(outlet)
        loss_out = mse(p_o, torch.zeros_like(p_o))

        topbot = make_inputs(int(cfg.batch_size.top_bot))
        side = torch.rand(topbot.shape[0], device=device) > 0.5
        topbot[side, 1] = y_bounds[1]
        topbot[~side, 1] = y_bounds[0]
        u_tb, v_tb, _ = model(topbot)
        loss_topbot = mse(u_tb, torch.full_like(u_tb, u_inlet)) + mse(v_tb, torch.zeros_like(v_tb))

        airfoil = make_inputs(int(cfg.batch_size.airfoil))
        airfoil[:, 0:1] = torch.empty_like(airfoil[:, 0:1]).uniform_(-0.1, 1.1)
        airfoil[:, 1:2] = torch.empty_like(airfoil[:, 1:2]).uniform_(-0.2, 0.2)
        u_a, v_a, _ = model(airfoil)
        loss_airfoil = mse(u_a, torch.zeros_like(u_a)) + mse(v_a, torch.zeros_like(v_a))

        loss = (
            float(cfg.loss.inlet) * loss_in
            + float(cfg.loss.outlet) * loss_out
            + float(cfg.loss.top_bot) * loss_topbot
            + float(cfg.loss.airfoil) * loss_airfoil
            + float(cfg.loss.interior_far_field) * loss_far
            + float(cfg.loss.interior_near_field) * loss_near
        )

        loss.backward()
        optimizer.step()
        scheduler.step()

        if step % int(cfg.training.log_every) == 0:
            print(
                f"step={step:06d} total={loss.item():.4e} inlet={loss_in.item():.4e} "
                f"outlet={loss_out.item():.4e} airfoil={loss_airfoil.item():.4e}"
            )


if __name__ == "__main__":
    run()
