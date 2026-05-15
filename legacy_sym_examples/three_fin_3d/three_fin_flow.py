# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

import hydra
import numpy as np
import torch
from omegaconf import DictConfig as PhysicsNeMoConfig
from torch import nn
from torch.optim import Adam, lr_scheduler


class FlowNet(nn.Module):
    def __init__(self, hidden_dim: int, num_layers: int):
        super().__init__()
        layers = [nn.Linear(3, hidden_dim), nn.Tanh()]
        for _ in range(max(0, num_layers - 1)):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
        layers.append(nn.Linear(hidden_dim, 4))
        self.net = nn.Sequential(*layers)

    def forward(self, xyz: torch.Tensor):
        out = self.net(xyz)
        return out[:, 0:1], out[:, 1:2], out[:, 2:3], out[:, 3:4]


def grad(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y), create_graph=True)[0]


def ns_residual(model: FlowNet, xyz: torch.Tensor, nu: float):
    xyz = xyz.requires_grad_(True)
    u, v, w, p = model(xyz)
    gu, gv, gw, gp = grad(u, xyz), grad(v, xyz), grad(w, xyz), grad(p, xyz)
    u_x, u_y, u_z = gu[:, 0:1], gu[:, 1:2], gu[:, 2:3]
    v_x, v_y, v_z = gv[:, 0:1], gv[:, 1:2], gv[:, 2:3]
    w_x, w_y, w_z = gw[:, 0:1], gw[:, 1:2], gw[:, 2:3]
    p_x, p_y, p_z = gp[:, 0:1], gp[:, 1:2], gp[:, 2:3]
    u_xx, u_yy, u_zz = grad(u_x, xyz)[:, 0:1], grad(u_y, xyz)[:, 1:2], grad(u_z, xyz)[:, 2:3]
    v_xx, v_yy, v_zz = grad(v_x, xyz)[:, 0:1], grad(v_y, xyz)[:, 1:2], grad(v_z, xyz)[:, 2:3]
    w_xx, w_yy, w_zz = grad(w_x, xyz)[:, 0:1], grad(w_y, xyz)[:, 1:2], grad(w_z, xyz)[:, 2:3]
    continuity = u_x + v_y + w_z
    mx = u * u_x + v * u_y + w * u_z + p_x - nu * (u_xx + u_yy + u_zz)
    my = u * v_x + v * v_y + w * v_z + p_y - nu * (v_xx + v_yy + v_zz)
    mz = u * w_x + v * w_y + w * w_z + p_z - nu * (w_xx + w_yy + w_zz)
    return continuity, mx, my, mz


def sample_box(n: int, x_bounds, y_bounds, z_bounds, device):
    x = torch.empty(n, 1, device=device).uniform_(x_bounds[0], x_bounds[1])
    y = torch.empty(n, 1, device=device).uniform_(y_bounds[0], y_bounds[1])
    z = torch.empty(n, 1, device=device).uniform_(z_bounds[0], z_bounds[1])
    return torch.cat([x, y, z], dim=1)


@hydra.main(version_base="1.3", config_path="conf", config_name="conf_flow")
def run(cfg: PhysicsNeMoConfig) -> None:
    torch.manual_seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x_bounds = (-2.5, 2.5)
    y_bounds = (-0.5, 0.5)
    z_bounds = (-0.5, 0.5)

    model = FlowNet(int(cfg.model.hidden_dim), int(cfg.model.num_layers)).to(device)
    optimizer = Adam(model.parameters(), lr=float(cfg.optimizer.lr))
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=float(cfg.scheduler.decay_rate))
    mse = nn.MSELoss()

    nu = float(cfg.physics.nu)
    inlet_u = float(cfg.physics.inlet_velocity)

    for step in range(int(cfg.training.max_steps)):
        optimizer.zero_grad()

        xyz_lr = sample_box(int(cfg.batch_size.InteriorLR), x_bounds, y_bounds, z_bounds, device)
        xyz_hr = sample_box(int(cfg.batch_size.InteriorHR), (-1.0, 1.0), y_bounds, z_bounds, device)
        cont_lr, mx_lr, my_lr, mz_lr = ns_residual(model, xyz_lr, nu)
        cont_hr, mx_hr, my_hr, mz_hr = ns_residual(model, xyz_hr, nu)
        z_lr = torch.zeros_like(cont_lr)
        z_hr = torch.zeros_like(cont_hr)
        loss_int = mse(cont_lr, z_lr) + mse(mx_lr, z_lr) + mse(my_lr, z_lr) + mse(mz_lr, z_lr)
        loss_int = loss_int + mse(cont_hr, z_hr) + mse(mx_hr, z_hr) + mse(my_hr, z_hr) + mse(mz_hr, z_hr)

        xyz_in = sample_box(int(cfg.batch_size.Inlet), (x_bounds[0], x_bounds[0]), y_bounds, z_bounds, device)
        xyz_in[:, 0:1] = x_bounds[0]
        u_in, v_in, w_in, _ = model(xyz_in)
        loss_inlet = mse(u_in, torch.full_like(u_in, inlet_u)) + mse(v_in, torch.zeros_like(v_in)) + mse(w_in, torch.zeros_like(w_in))

        xyz_out = sample_box(int(cfg.batch_size.Outlet), (x_bounds[1], x_bounds[1]), y_bounds, z_bounds, device)
        xyz_out[:, 0:1] = x_bounds[1]
        _, _, _, p_out = model(xyz_out)
        loss_outlet = mse(p_out, torch.zeros_like(p_out))

        xyz_wall = sample_box(int(cfg.batch_size.NoSlip), x_bounds, y_bounds, z_bounds, device)
        sel = torch.randint(0, 4, (xyz_wall.shape[0],), device=device)
        xyz_wall[sel == 0, 1] = y_bounds[0]
        xyz_wall[sel == 1, 1] = y_bounds[1]
        xyz_wall[sel == 2, 2] = z_bounds[0]
        xyz_wall[sel == 3, 2] = z_bounds[1]
        u_w, v_w, w_w, _ = model(xyz_wall)
        loss_wall = mse(u_w, torch.zeros_like(u_w)) + mse(v_w, torch.zeros_like(v_w)) + mse(w_w, torch.zeros_like(w_w))

        loss = (
            float(cfg.loss.Inlet) * loss_inlet
            + float(cfg.loss.Outlet) * loss_outlet
            + float(cfg.loss.NoSlip) * loss_wall
            + float(cfg.loss.InteriorLR) * loss_int
        )

        loss.backward()
        optimizer.step()
        scheduler.step()

        if step % int(cfg.training.log_every) == 0:
            print(f"step={step:06d} total={loss.item():.4e} inlet={loss_inlet.item():.4e} outlet={loss_outlet.item():.4e}")


if __name__ == "__main__":
    run()
