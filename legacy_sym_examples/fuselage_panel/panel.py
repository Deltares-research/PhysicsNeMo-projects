# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

import hydra
import numpy as np
import torch
from omegaconf import DictConfig as PhysicsNeMoConfig
from torch import nn
from torch.optim import Adam, lr_scheduler


class ElasticNet2D(nn.Module):
    def __init__(self, hidden_dim: int, num_layers: int):
        super().__init__()
        layers = [nn.Linear(2, hidden_dim), nn.Tanh()]
        for _ in range(max(0, num_layers - 1)):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
        layers.append(nn.Linear(hidden_dim, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, xy: torch.Tensor):
        out = self.net(xy)
        return out[:, 0:1], out[:, 1:2]


def grad(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y), create_graph=True)[0]


def elasticity_residual(model: ElasticNet2D, xy: torch.Tensor):
    xy = xy.requires_grad_(True)
    ux, uy = model(xy)
    ux_x = grad(ux, xy)[:, 0:1]
    uy_y = grad(uy, xy)[:, 1:2]
    div_u = ux_x + uy_y
    rx = grad(div_u, xy)[:, 0:1]
    ry = grad(div_u, xy)[:, 1:2]
    return rx, ry


def sample_rect(n: int, x_bounds, y_bounds, device):
    x = torch.empty(n, 1, device=device).uniform_(x_bounds[0], x_bounds[1])
    y = torch.empty(n, 1, device=device).uniform_(y_bounds[0], y_bounds[1])
    return torch.cat([x, y], dim=1)


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def run(cfg: PhysicsNeMoConfig) -> None:
    torch.manual_seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x_bounds = (-1.0, 1.0)
    y_bounds = (-0.5, 0.5)

    model = ElasticNet2D(int(cfg.model.hidden_dim), int(cfg.model.num_layers)).to(device)
    optimizer = Adam(model.parameters(), lr=float(cfg.optimizer.lr))
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=float(cfg.scheduler.decay_rate))
    mse = nn.MSELoss()

    for step in range(int(cfg.training.max_steps)):
        optimizer.zero_grad()

        xy_lr = sample_rect(int(cfg.batch_size.lr_interior), x_bounds, y_bounds, device)
        xy_hr = sample_rect(int(cfg.batch_size.hr_interior), (-0.3, 0.3), y_bounds, device)
        rx1, ry1 = elasticity_residual(model, xy_lr)
        rx2, ry2 = elasticity_residual(model, xy_hr)
        z1 = torch.zeros_like(rx1)
        z2 = torch.zeros_like(rx2)
        loss_int = mse(rx1, z1) + mse(ry1, z1) + mse(rx2, z2) + mse(ry2, z2)

        left = sample_rect(int(cfg.batch_size.panel_left), (x_bounds[0], x_bounds[0]), y_bounds, device)
        left[:, 0:1] = x_bounds[0]
        ux_l, uy_l = model(left)
        loss_left = mse(ux_l, torch.zeros_like(ux_l)) + mse(uy_l, torch.zeros_like(uy_l))

        right = sample_rect(int(cfg.batch_size.panel_right), (x_bounds[1], x_bounds[1]), y_bounds, device)
        right[:, 0:1] = x_bounds[1]
        ux_r, _ = model(right)
        loss_right = mse(ux_r, torch.full_like(ux_r, float(cfg.physics.right_disp)))

        bottom = sample_rect(int(cfg.batch_size.panel_bottom), x_bounds, (y_bounds[0], y_bounds[0]), device)
        bottom[:, 1:2] = y_bounds[0]
        _, uy_b = model(bottom)
        loss_bottom = mse(uy_b, torch.zeros_like(uy_b))

        top = sample_rect(int(cfg.batch_size.panel_top), x_bounds, (y_bounds[1], y_bounds[1]), device)
        top[:, 1:2] = y_bounds[1]
        _, uy_t = model(top)
        loss_top = mse(uy_t, torch.zeros_like(uy_t))

        corner = sample_rect(int(cfg.batch_size.panel_corner), (x_bounds[0], x_bounds[0]), (y_bounds[0], y_bounds[0]), device)
        corner[:, 0:1] = x_bounds[0]
        corner[:, 1:2] = y_bounds[0]
        ux_c, uy_c = model(corner)
        loss_corner = mse(ux_c, torch.zeros_like(ux_c)) + mse(uy_c, torch.zeros_like(uy_c))

        window = sample_rect(int(cfg.batch_size.panel_window), (-0.2, 0.2), (-0.2, 0.2), device)
        ux_w, uy_w = model(window)
        loss_window = mse(ux_w, torch.zeros_like(ux_w)) + mse(uy_w, torch.zeros_like(uy_w))

        loss = (
            float(cfg.loss.lr_interior) * loss_int
            + float(cfg.loss.panel_left) * loss_left
            + float(cfg.loss.panel_right) * loss_right
            + float(cfg.loss.panel_bottom) * loss_bottom
            + float(cfg.loss.panel_top) * loss_top
            + float(cfg.loss.panel_corner) * loss_corner
            + float(cfg.loss.panel_window) * loss_window
        )

        loss.backward()
        optimizer.step()
        scheduler.step()

        if step % int(cfg.training.log_every) == 0:
            print(
                f"step={step:06d} total={loss.item():.4e} interior={loss_int.item():.4e} "
                f"left={loss_left.item():.4e} right={loss_right.item():.4e}"
            )


if __name__ == "__main__":
    run()
