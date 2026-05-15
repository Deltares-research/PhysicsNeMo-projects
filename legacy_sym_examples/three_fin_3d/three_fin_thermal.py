# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

import hydra
import numpy as np
import torch
from omegaconf import DictConfig as PhysicsNeMoConfig
from torch import nn
from torch.optim import Adam, lr_scheduler


class HeatNet(nn.Module):
    def __init__(self, hidden_dim: int, num_layers: int):
        super().__init__()
        layers = [nn.Linear(3, hidden_dim), nn.Tanh()]
        for _ in range(max(0, num_layers - 1)):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, xyz: torch.Tensor):
        return self.net(xyz)


def grad(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y), create_graph=True)[0]


def diffusion_residual(model: HeatNet, xyz: torch.Tensor, d: float):
    xyz = xyz.requires_grad_(True)
    t = model(xyz)
    g = grad(t, xyz)
    t_x, t_y, t_z = g[:, 0:1], g[:, 1:2], g[:, 2:3]
    t_xx = grad(t_x, xyz)[:, 0:1]
    t_yy = grad(t_y, xyz)[:, 1:2]
    t_zz = grad(t_z, xyz)[:, 2:3]
    return d * (t_xx + t_yy + t_zz)


def sample_box(n: int, x_bounds, y_bounds, z_bounds, device):
    x = torch.empty(n, 1, device=device).uniform_(x_bounds[0], x_bounds[1])
    y = torch.empty(n, 1, device=device).uniform_(y_bounds[0], y_bounds[1])
    z = torch.empty(n, 1, device=device).uniform_(z_bounds[0], z_bounds[1])
    return torch.cat([x, y, z], dim=1)


@hydra.main(version_base="1.3", config_path="conf", config_name="conf_thermal")
def run(cfg: PhysicsNeMoConfig) -> None:
    torch.manual_seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x_bounds = (-2.5, 2.5)
    y_bounds = (-0.5, 0.5)
    z_bounds = (-0.5, 0.5)
    fin_x = (-0.5, 0.5)
    fin_y = (-0.5, -0.2)
    fin_z = (-0.25, 0.25)

    fluid = HeatNet(int(cfg.model.hidden_dim), int(cfg.model.num_layers)).to(device)
    solid = HeatNet(int(cfg.model.hidden_dim), int(cfg.model.num_layers)).to(device)
    optimizer = Adam(list(fluid.parameters()) + list(solid.parameters()), lr=float(cfg.optimizer.lr))
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=float(cfg.scheduler.decay_rate))
    mse = nn.MSELoss()

    inlet_t = float(cfg.physics.inlet_temp)
    grad_t = float(cfg.physics.source_grad)
    d_f = float(cfg.physics.diffusivity_fluid)
    d_s = float(cfg.physics.diffusivity_solid)

    for step in range(int(cfg.training.max_steps)):
        optimizer.zero_grad()

        xyz_fl_lr = sample_box(int(cfg.batch_size.InteriorLR), x_bounds, y_bounds, z_bounds, device)
        xyz_fl_hr = sample_box(int(cfg.batch_size.InteriorHR), (-1.0, 1.0), y_bounds, z_bounds, device)
        xyz_s = sample_box(int(cfg.batch_size.SolidInterior), fin_x, fin_y, fin_z, device)
        loss_fl = mse(diffusion_residual(fluid, xyz_fl_lr, d_f), torch.zeros(int(cfg.batch_size.InteriorLR), 1, device=device))
        loss_fl = loss_fl + mse(diffusion_residual(fluid, xyz_fl_hr, d_f), torch.zeros(int(cfg.batch_size.InteriorHR), 1, device=device))
        loss_s = mse(diffusion_residual(solid, xyz_s, d_s), torch.zeros(int(cfg.batch_size.SolidInterior), 1, device=device))

        xyz_in = sample_box(int(cfg.batch_size.Inlet), (x_bounds[0], x_bounds[0]), y_bounds, z_bounds, device)
        xyz_in[:, 0:1] = x_bounds[0]
        t_in = fluid(xyz_in)
        loss_in = mse(t_in, torch.full_like(t_in, inlet_t))

        xyz_out = sample_box(int(cfg.batch_size.Outlet), (x_bounds[1], x_bounds[1]), y_bounds, z_bounds, device)
        xyz_out[:, 0:1] = x_bounds[1]
        xyz_out = xyz_out.requires_grad_(True)
        t_out = fluid(xyz_out)
        g_out = grad(t_out, xyz_out)
        loss_out = mse(g_out[:, 0:1], torch.zeros_like(g_out[:, 0:1]))

        xyz_if = sample_box(int(cfg.batch_size.SolidInterface), fin_x, fin_y, fin_z, device)
        loss_if = mse(fluid(xyz_if), solid(xyz_if))

        xyz_src = sample_box(int(cfg.batch_size.HeatSource), (fin_x[0] + 0.1, fin_x[1] - 0.1), (fin_y[0], fin_y[0]), (fin_z[0] + 0.05, fin_z[1] - 0.05), device)
        xyz_src[:, 1:2] = fin_y[0]
        xyz_src = xyz_src.requires_grad_(True)
        ts = solid(xyz_src)
        gs = grad(ts, xyz_src)
        loss_src = mse(gs[:, 1:2], torch.full_like(gs[:, 1:2], grad_t))

        loss = (
            float(cfg.loss.Inlet) * loss_in
            + float(cfg.loss.Outlet) * loss_out
            + float(cfg.loss.SolidInterface) * loss_if
            + float(cfg.loss.HeatSource) * loss_src
            + float(cfg.loss.InteriorLR) * loss_fl
            + float(cfg.loss.SolidInterior) * loss_s
        )

        loss.backward()
        optimizer.step()
        scheduler.step()

        if step % int(cfg.training.log_every) == 0:
            print(f"step={step:06d} total={loss.item():.4e} inlet={loss_in.item():.4e} outlet={loss_out.item():.4e}")


if __name__ == "__main__":
    run()
