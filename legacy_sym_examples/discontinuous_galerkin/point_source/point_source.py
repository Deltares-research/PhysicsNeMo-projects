# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

import hydra
import numpy as np
import torch
from omegaconf import DictConfig as PhysicsNeMoConfig
from torch import nn
from torch.optim import Adam, lr_scheduler


class DGNet(nn.Module):
    def __init__(self, in_features: int, out_features: int, hidden_dim: int, num_layers: int):
        super().__init__()
        layers = [nn.Linear(in_features, hidden_dim), nn.Tanh()]
        for _ in range(max(0, num_layers - 1)):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
        layers.append(nn.Linear(hidden_dim, out_features))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def sample_interior(n: int, device: torch.device) -> torch.Tensor:
    xy = torch.rand(n, 2, device=device)
    return -0.5 + xy


def sample_boundary(n: int, device: torch.device) -> torch.Tensor:
    n_side = max(1, n // 4)
    t = -0.5 + torch.rand(n_side, 1, device=device)
    left = torch.cat([torch.full_like(t, -0.5), t], dim=1)
    right = torch.cat([torch.full_like(t, 0.5), t], dim=1)
    bottom = torch.cat([t, torch.full_like(t, -0.5)], dim=1)
    top = torch.cat([t, torch.full_like(t, 0.5)], dim=1)
    return torch.cat([left, right, bottom, top], dim=0)


def diffusion_residual(model: nn.Module, xy: torch.Tensor, sigma: float) -> torch.Tensor:
    xy = xy.requires_grad_(True)
    u = model(xy)
    gu = torch.autograd.grad(u, xy, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_x = gu[:, 0:1]
    u_y = gu[:, 1:2]
    u_xx = torch.autograd.grad(u_x, xy, grad_outputs=torch.ones_like(u_x), create_graph=True)[0][:, 0:1]
    u_yy = torch.autograd.grad(u_y, xy, grad_outputs=torch.ones_like(u_y), create_graph=True)[0][:, 1:2]

    x = xy[:, 0:1]
    y = xy[:, 1:2]
    norm2 = x * x + y * y
    source = torch.exp(-norm2 / (2.0 * sigma * sigma)) / (2.0 * torch.pi * sigma * sigma)
    return u_xx + u_yy + source


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def run(cfg: PhysicsNeMoConfig) -> None:
    torch.manual_seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DGNet(2, 1, int(cfg.model.hidden_dim), int(cfg.model.num_layers)).to(device)
    mse = nn.MSELoss(reduction="mean")
    optimizer = Adam(model.parameters(), lr=float(cfg.optimizer.lr))
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=float(cfg.scheduler.decay_rate))

    sigma = float(cfg.physics.source_sigma)

    for step in range(int(cfg.training.max_steps)):
        optimizer.zero_grad()

        boundary = sample_boundary(int(cfg.batch_size.boundary), device)
        interior = sample_interior(int(cfg.batch_size.interior), device)

        u_b = model(boundary)
        loss_b = mse(u_b, torch.zeros_like(u_b))

        res = diffusion_residual(model, interior, sigma)
        loss_i = mse(res, torch.zeros_like(res))

        loss = float(cfg.loss.boundary) * loss_b + float(cfg.loss.interior) * loss_i
        loss.backward()
        optimizer.step()
        scheduler.step()

        if step % int(cfg.training.log_every) == 0:
            with torch.no_grad():
                c = model(torch.zeros(1, 2, device=device))
                print(
                    f"step={step:06d} total={loss.item():.4e} boundary={loss_b.item():.4e} "
                    f"interior={loss_i.item():.4e} u_center={c.item():.4e}"
                )


if __name__ == "__main__":
    run()
