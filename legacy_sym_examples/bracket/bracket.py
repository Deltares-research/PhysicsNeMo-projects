# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

import hydra
import numpy as np
import torch
from omegaconf import DictConfig as PhysicsNeMoConfig
from torch import nn
from torch.optim import Adam, lr_scheduler


class ElasticNet3D(nn.Module):
    def __init__(self, hidden_dim: int, num_layers: int):
        super().__init__()
        layers = [nn.Linear(3, hidden_dim), nn.Tanh()]
        for _ in range(max(0, num_layers - 1)):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
        layers.append(nn.Linear(hidden_dim, 3))
        self.net = nn.Sequential(*layers)

    def forward(self, xyz: torch.Tensor):
        out = self.net(xyz)
        return out[:, 0:1], out[:, 1:2], out[:, 2:3]


def grad(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y), create_graph=True)[0]


def elasticity_residual(model: ElasticNet3D, xyz: torch.Tensor):
    xyz = xyz.requires_grad_(True)
    ux, uy, uz = model(xyz)

    ux_x = grad(ux, xyz)[:, 0:1]
    ux_y = grad(ux, xyz)[:, 1:2]
    ux_z = grad(ux, xyz)[:, 2:3]
    uy_x = grad(uy, xyz)[:, 0:1]
    uy_y = grad(uy, xyz)[:, 1:2]
    uy_z = grad(uy, xyz)[:, 2:3]
    uz_x = grad(uz, xyz)[:, 0:1]
    uz_y = grad(uz, xyz)[:, 1:2]
    uz_z = grad(uz, xyz)[:, 2:3]

    div_u = ux_x + uy_y + uz_z
    rx = grad(div_u, xyz)[:, 0:1]
    ry = grad(div_u, xyz)[:, 1:2]
    rz = grad(div_u, xyz)[:, 2:3]
    return rx, ry, rz


def sample_box(n: int, x_bounds, y_bounds, z_bounds, device):
    x = torch.empty(n, 1, device=device).uniform_(x_bounds[0], x_bounds[1])
    y = torch.empty(n, 1, device=device).uniform_(y_bounds[0], y_bounds[1])
    z = torch.empty(n, 1, device=device).uniform_(z_bounds[0], z_bounds[1])
    return torch.cat([x, y, z], dim=1)


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def run(cfg: PhysicsNeMoConfig) -> None:
    torch.manual_seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x_bounds = (-1.0, 1.0)
    y_bounds = (-1.0, 1.0)
    z_bounds = (-1.0, 1.0)

    model = ElasticNet3D(int(cfg.model.hidden_dim), int(cfg.model.num_layers)).to(device)
    optimizer = Adam(model.parameters(), lr=float(cfg.optimizer.lr))
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=float(cfg.scheduler.decay_rate))
    mse = nn.MSELoss()

    for step in range(int(cfg.training.max_steps)):
        optimizer.zero_grad()

        xyz_support = sample_box(int(cfg.batch_size.interior_support), x_bounds, y_bounds, z_bounds, device)
        xyz_bracket = sample_box(int(cfg.batch_size.interior_bracket), (-0.6, 0.6), y_bounds, z_bounds, device)
        rx1, ry1, rz1 = elasticity_residual(model, xyz_support)
        rx2, ry2, rz2 = elasticity_residual(model, xyz_bracket)
        z1 = torch.zeros_like(rx1)
        z2 = torch.zeros_like(rx2)
        loss_int = mse(rx1, z1) + mse(ry1, z1) + mse(rz1, z1) + mse(rx2, z2) + mse(ry2, z2) + mse(rz2, z2)

        xyz_back = sample_box(int(cfg.batch_size.backBC), (x_bounds[0], x_bounds[0]), y_bounds, z_bounds, device)
        xyz_back[:, 0:1] = x_bounds[0]
        ux_b, uy_b, uz_b = model(xyz_back)
        loss_back = mse(ux_b, torch.zeros_like(ux_b)) + mse(uy_b, torch.zeros_like(uy_b)) + mse(uz_b, torch.zeros_like(uz_b))

        xyz_front = sample_box(int(cfg.batch_size.frontBC), (x_bounds[1], x_bounds[1]), y_bounds, z_bounds, device)
        xyz_front[:, 0:1] = x_bounds[1]
        ux_f, _, _ = model(xyz_front)
        loss_front = mse(ux_f, torch.full_like(ux_f, float(cfg.physics.front_disp)))

        xyz_surface = sample_box(int(cfg.batch_size.surfaceBC), x_bounds, y_bounds, z_bounds, device)
        sel = torch.randint(0, 4, (xyz_surface.shape[0],), device=device)
        xyz_surface[sel == 0, 1] = y_bounds[0]
        xyz_surface[sel == 1, 1] = y_bounds[1]
        xyz_surface[sel == 2, 2] = z_bounds[0]
        xyz_surface[sel == 3, 2] = z_bounds[1]
        _, uy_s, uz_s = model(xyz_surface)
        loss_surface = mse(uy_s, torch.zeros_like(uy_s)) + mse(uz_s, torch.zeros_like(uz_s))

        loss = (
            float(cfg.loss.interior) * loss_int
            + float(cfg.loss.backBC) * loss_back
            + float(cfg.loss.frontBC) * loss_front
            + float(cfg.loss.surfaceBC) * loss_surface
        )

        loss.backward()
        optimizer.step()
        scheduler.step()

        if step % int(cfg.training.log_every) == 0:
            print(
                f"step={step:06d} total={loss.item():.4e} interior={loss_int.item():.4e} "
                f"back={loss_back.item():.4e} front={loss_front.item():.4e} surface={loss_surface.item():.4e}"
            )


if __name__ == "__main__":
    run()
