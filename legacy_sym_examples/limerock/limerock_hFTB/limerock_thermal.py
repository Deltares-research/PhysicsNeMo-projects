# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

import hydra
import numpy as np
import torch
from omegaconf import DictConfig as PhysicsNeMoConfig
from torch import nn
from torch.optim import Adam, lr_scheduler


class ThermalNet(nn.Module):
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


def diffusion_residual(model: ThermalNet, xyz: torch.Tensor, d: float):
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

    x_bounds = (-1.5, 1.5)
    y_bounds = (-0.5, 0.5)
    z_bounds = (-0.5, 0.5)
    solid_x = (-0.4, 0.4)
    solid_y = (-0.5, -0.2)
    solid_z = (-0.2, 0.2)

    fluid = ThermalNet(int(cfg.model.hidden_dim), int(cfg.model.num_layers)).to(device)
    solid = ThermalNet(int(cfg.model.hidden_dim), int(cfg.model.num_layers)).to(device)
    optimizer = Adam(list(fluid.parameters()) + list(solid.parameters()), lr=float(cfg.optimizer.lr))
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=float(cfg.scheduler.decay_rate))
    mse = nn.MSELoss()

    inlet_temp = float(cfg.physics.inlet_temp)
    d_f = float(cfg.physics.diffusivity_fluid)
    d_s = float(cfg.physics.diffusivity_solid)
    base_grad = float(cfg.physics.base_gradient)

    for step in range(int(cfg.training.max_steps)):
        optimizer.zero_grad()

        lr_f = sample_box(int(cfg.batch_size.lr_interior_f), x_bounds, y_bounds, z_bounds, device)
        hr_f = sample_box(int(cfg.batch_size.hr_interior_f), (-0.8, 0.8), y_bounds, z_bounds, device)
        in_s = sample_box(int(cfg.batch_size.interior_s), solid_x, solid_y, solid_z, device)
        loss_lr = mse(diffusion_residual(fluid, lr_f, d_f), torch.zeros(int(cfg.batch_size.lr_interior_f), 1, device=device))
        loss_hr = mse(diffusion_residual(fluid, hr_f, d_f), torch.zeros(int(cfg.batch_size.hr_interior_f), 1, device=device))
        loss_s = mse(diffusion_residual(solid, in_s, d_s), torch.zeros(int(cfg.batch_size.interior_s), 1, device=device))

        inlet = sample_box(int(cfg.batch_size.inlet), (x_bounds[0], x_bounds[0]), y_bounds, z_bounds, device)
        inlet[:, 0:1] = x_bounds[0]
        ti = fluid(inlet)
        loss_in = mse(ti, torch.full_like(ti, inlet_temp))

        outlet = sample_box(int(cfg.batch_size.outlet), (x_bounds[1], x_bounds[1]), y_bounds, z_bounds, device)
        outlet[:, 0:1] = x_bounds[1]
        outlet = outlet.requires_grad_(True)
        to = fluid(outlet)
        go = grad(to, outlet)
        loss_out = mse(go[:, 0:1], torch.zeros_like(go[:, 0:1]))

        walls = sample_box(int(cfg.batch_size.no_slip), x_bounds, y_bounds, z_bounds, device)
        sel = torch.randint(0, 4, (walls.shape[0],), device=device)
        walls[sel == 0, 1] = y_bounds[0]
        walls[sel == 1, 1] = y_bounds[1]
        walls[sel == 2, 2] = z_bounds[0]
        walls[sel == 3, 2] = z_bounds[1]
        walls = walls.requires_grad_(True)
        tw = fluid(walls)
        gw = grad(tw, walls)
        loss_walls = mse(gw[:, 1:2], torch.zeros_like(gw[:, 1:2])) + mse(gw[:, 2:3], torch.zeros_like(gw[:, 2:3]))

        interface = sample_box(int(cfg.batch_size.interface), solid_x, solid_y, solid_z, device)
        loss_if = mse(fluid(interface), solid(interface))

        base = sample_box(int(cfg.batch_size.base), (solid_x[0] + 0.05, solid_x[1] - 0.05), (solid_y[0], solid_y[0]), (solid_z[0] + 0.05, solid_z[1] - 0.05), device)
        base[:, 1:2] = solid_y[0]
        base = base.requires_grad_(True)
        tb = solid(base)
        gb = grad(tb, base)
        loss_base = mse(gb[:, 1:2], torch.full_like(gb[:, 1:2], base_grad))

        loss = (
            float(cfg.loss.inlet) * loss_in
            + float(cfg.loss.outlet) * loss_out
            + float(cfg.loss.no_slip) * loss_walls
            + float(cfg.loss.lr_interior_f) * loss_lr
            + float(cfg.loss.hr_interior_f) * loss_hr
            + float(cfg.loss.interior_s) * loss_s
            + float(cfg.loss.interface) * loss_if
            + float(cfg.loss.base) * loss_base
        )

        loss.backward()
        optimizer.step()
        scheduler.step()

        if step % int(cfg.training.log_every) == 0:
            print(
                f"step={step:06d} total={loss.item():.4e} inlet={loss_in.item():.4e} "
                f"outlet={loss_out.item():.4e} interface={loss_if.item():.4e}"
            )


if __name__ == "__main__":
    run()
