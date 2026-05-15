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


def diffusion_residual(model: HeatNet, xyz: torch.Tensor, diffusivity: float):
    xyz = xyz.requires_grad_(True)
    t = model(xyz)
    g = grad(t, xyz)
    t_x, t_y, t_z = g[:, 0:1], g[:, 1:2], g[:, 2:3]
    t_xx = grad(t_x, xyz)[:, 0:1]
    t_yy = grad(t_y, xyz)[:, 1:2]
    t_zz = grad(t_z, xyz)[:, 2:3]
    return diffusivity * (t_xx + t_yy + t_zz)


def sample_box(n: int, x_bounds, y_bounds, z_bounds, device):
    x = torch.empty(n, 1, device=device).uniform_(x_bounds[0], x_bounds[1])
    y = torch.empty(n, 1, device=device).uniform_(y_bounds[0], y_bounds[1])
    z = torch.empty(n, 1, device=device).uniform_(z_bounds[0], z_bounds[1])
    return torch.cat([x, y, z], dim=1)


def sample_chip_surface(n: int, chip_x, chip_y, chip_z, device):
    pts = sample_box(n, chip_x, chip_y, chip_z, device)
    face = torch.randint(0, 6, (n,), device=device)
    pts[face == 0, 0] = chip_x[0]
    pts[face == 1, 0] = chip_x[1]
    pts[face == 2, 1] = chip_y[0]
    pts[face == 3, 1] = chip_y[1]
    pts[face == 4, 2] = chip_z[0]
    pts[face == 5, 2] = chip_z[1]
    return pts


@hydra.main(version_base="1.3", config_path="conf_heat", config_name="config")
def run(cfg: PhysicsNeMoConfig) -> None:
    torch.manual_seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x_bounds = (-2.5, 2.5)
    y_bounds = (-0.5, 0.5)
    z_bounds = (-0.5, 0.5)
    chip_x = (-0.5, 0.5)
    chip_y = (-0.5, -0.3)
    chip_z = (-0.2, 0.2)

    fluid_net = HeatNet(int(cfg.model.hidden_dim), int(cfg.model.num_layers)).to(device)
    solid_net = HeatNet(int(cfg.model.hidden_dim), int(cfg.model.num_layers)).to(device)
    optimizer = Adam(list(fluid_net.parameters()) + list(solid_net.parameters()), lr=float(cfg.optimizer.lr))
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=float(cfg.scheduler.decay_rate))
    mse = nn.MSELoss()

    inlet_temp = float(cfg.physics.inlet_temp)
    d_fluid = float(cfg.physics.diffusivity_fluid)
    d_solid = float(cfg.physics.diffusivity_solid)
    source_grad = float(cfg.physics.source_grad)

    for step in range(int(cfg.training.max_steps)):
        optimizer.zero_grad()

        xyz_fl_lr = sample_box(int(cfg.batch_size.lr_flow_interior), x_bounds, y_bounds, z_bounds, device)
        res_fl_lr = diffusion_residual(fluid_net, xyz_fl_lr, d_fluid)
        loss_fl_lr = mse(res_fl_lr, torch.zeros_like(res_fl_lr))

        xyz_fl_hr = sample_box(int(cfg.batch_size.hr_flow_interior), (-1.0, 1.0), y_bounds, z_bounds, device)
        res_fl_hr = diffusion_residual(fluid_net, xyz_fl_hr, d_fluid)
        loss_fl_hr = mse(res_fl_hr, torch.zeros_like(res_fl_hr))

        xyz_s = sample_box(int(cfg.batch_size.solid_interior), chip_x, chip_y, chip_z, device)
        res_s = diffusion_residual(solid_net, xyz_s, d_solid)
        loss_s = mse(res_s, torch.zeros_like(res_s))

        yz_in = sample_box(int(cfg.batch_size.inlet), (0.0, 0.0), y_bounds, z_bounds, device)
        yz_in[:, 0:1] = x_bounds[0]
        t_in = fluid_net(yz_in)
        loss_inlet = mse(t_in, torch.full_like(t_in, inlet_temp))

        yz_out = sample_box(int(cfg.batch_size.outlet), (0.0, 0.0), y_bounds, z_bounds, device)
        yz_out[:, 0:1] = x_bounds[1]
        yz_out = yz_out.requires_grad_(True)
        t_out = fluid_net(yz_out)
        g_out = grad(t_out, yz_out)
        loss_outlet = mse(g_out[:, 0:1], torch.zeros_like(g_out[:, 0:1]))

        xyz_wall = sample_box(int(cfg.batch_size.channel_walls), x_bounds, y_bounds, z_bounds, device)
        wsel = torch.randint(0, 4, (xyz_wall.shape[0],), device=device)
        xyz_wall[wsel == 0, 1] = y_bounds[0]
        xyz_wall[wsel == 1, 1] = y_bounds[1]
        xyz_wall[wsel == 2, 2] = z_bounds[0]
        xyz_wall[wsel == 3, 2] = z_bounds[1]
        xyz_wall = xyz_wall.requires_grad_(True)
        t_wall = fluid_net(xyz_wall)
        g_wall = grad(t_wall, xyz_wall)
        loss_wall = mse(g_wall[:, 1:2], torch.zeros_like(g_wall[:, 1:2])) + mse(g_wall[:, 2:3], torch.zeros_like(g_wall[:, 2:3]))

        xyz_if = sample_chip_surface(int(cfg.batch_size.fluid_solid_interface), chip_x, chip_y, chip_z, device)
        t_fi = fluid_net(xyz_if)
        t_si = solid_net(xyz_if)
        loss_if = mse(t_fi, t_si)

        xyz_src = sample_box(int(cfg.batch_size.heat_source), (chip_x[0] + 0.1, chip_x[1] - 0.1), (chip_y[0], chip_y[0]), (chip_z[0] + 0.05, chip_z[1] - 0.05), device)
        xyz_src[:, 1:2] = chip_y[0]
        xyz_src = xyz_src.requires_grad_(True)
        t_src = solid_net(xyz_src)
        g_src = grad(t_src, xyz_src)
        loss_src = mse(g_src[:, 1:2], torch.full_like(g_src[:, 1:2], source_grad))

        loss = (
            float(cfg.loss.inlet) * loss_inlet
            + float(cfg.loss.outlet) * loss_outlet
            + float(cfg.loss.channel_walls) * loss_wall
            + float(cfg.loss.fluid_solid_interface) * loss_if
            + float(cfg.loss.heat_source) * loss_src
            + float(cfg.loss.lr_flow_interior) * loss_fl_lr
            + float(cfg.loss.hr_flow_interior) * loss_fl_hr
            + float(cfg.loss.solid_interior) * loss_s
        )

        loss.backward()
        optimizer.step()
        scheduler.step()

        if step % int(cfg.training.log_every) == 0:
            print(
                f"step={step:06d} total={loss.item():.4e} inlet={loss_inlet.item():.4e} "
                f"outlet={loss_outlet.item():.4e} if={loss_if.item():.4e} src={loss_src.item():.4e}"
            )


if __name__ == "__main__":
    run()
