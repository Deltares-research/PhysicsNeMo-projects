# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

import hydra
import numpy as np
import torch
from omegaconf import DictConfig as PhysicsNeMoConfig
from torch import nn
from torch.optim import Adam, lr_scheduler


class HelmholtzNet(nn.Module):
    def __init__(self, in_features: int, out_features: int, hidden_dim: int, num_layers: int):
        super().__init__()
        layers = [nn.Linear(in_features, hidden_dim), nn.Tanh()]
        for _ in range(max(0, num_layers - 1)):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
        layers.append(nn.Linear(hidden_dim, out_features))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def exact_solution(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.sin(torch.pi * x) * torch.sin(4.0 * torch.pi * y)


def forcing_term(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    u = exact_solution(x, y)
    return (1.0 - 17.0 * torch.pi**2) * u


def pde_residual(model: nn.Module, coords: torch.Tensor) -> torch.Tensor:
    coords = coords.requires_grad_(True)
    pred = model(coords)
    grad = torch.autograd.grad(pred, coords, grad_outputs=torch.ones_like(pred), create_graph=True)[0]
    u_x = grad[:, 0:1]
    u_y = grad[:, 1:2]

    u_xx = torch.autograd.grad(u_x, coords, grad_outputs=torch.ones_like(u_x), create_graph=True)[0][:, 0:1]
    u_yy = torch.autograd.grad(u_y, coords, grad_outputs=torch.ones_like(u_y), create_graph=True)[0][:, 1:2]

    x = coords[:, 0:1]
    y = coords[:, 1:2]
    f = forcing_term(x, y)
    return u_xx + u_yy + pred - f


def sample_interior(n: int, device: torch.device) -> torch.Tensor:
    xy = torch.rand(n, 2, device=device)
    return -1.0 + 2.0 * xy


def sample_boundary(n: int, device: torch.device) -> torch.Tensor:
    n_side = max(1, n // 4)
    t = -1.0 + 2.0 * torch.rand(n_side, 1, device=device)
    left = torch.cat([torch.full_like(t, -1.0), t], dim=1)
    right = torch.cat([torch.full_like(t, 1.0), t], dim=1)
    bottom = torch.cat([t, torch.full_like(t, -1.0)], dim=1)
    top = torch.cat([t, torch.full_like(t, 1.0)], dim=1)
    return torch.cat([left, right, bottom, top], dim=0)


def maybe_load_validation(device: torch.device) -> tuple[torch.Tensor, torch.Tensor] | None:
    file_path = "validation/helmholtz.csv"
    if not os.path.exists(file_path):
        return None

    raw = np.genfromtxt(file_path, delimiter=",", names=True)
    if raw is None:
        return None

    x = np.asarray(raw["x"], dtype=np.float32).reshape(-1, 1)
    y = np.asarray(raw["y"], dtype=np.float32).reshape(-1, 1)
    u = np.asarray(raw["z"], dtype=np.float32).reshape(-1, 1)

    xy = torch.from_numpy(np.concatenate([x, y], axis=1)).to(device)
    uu = torch.from_numpy(u).to(device)
    return xy, uu


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def run(cfg: PhysicsNeMoConfig) -> None:
    torch.manual_seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = HelmholtzNet(
        in_features=2,
        out_features=1,
        hidden_dim=int(cfg.model.hidden_dim),
        num_layers=int(cfg.model.num_layers),
    ).to(device)

    optimizer = Adam(model.parameters(), lr=float(cfg.optimizer.lr))
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=float(cfg.scheduler.decay_rate))
    loss_fn = nn.MSELoss(reduction="mean")

    val_data = maybe_load_validation(device)

    for step in range(int(cfg.training.max_steps)):
        optimizer.zero_grad()

        wall_pts = sample_boundary(int(cfg.batch_size.wall), device)
        int_pts = sample_interior(int(cfg.batch_size.interior), device)

        u_wall = model(wall_pts)
        loss_wall = loss_fn(u_wall, torch.zeros_like(u_wall))

        resid = pde_residual(model, int_pts)
        loss_pde = loss_fn(resid, torch.zeros_like(resid))

        loss = float(cfg.loss.wall) * loss_wall + float(cfg.loss.interior) * loss_pde
        loss.backward()
        optimizer.step()
        scheduler.step()

        if step % int(cfg.training.log_every) == 0:
            msg = (
                f"step={step:06d} total={loss.item():.4e} "
                f"wall={loss_wall.item():.4e} interior={loss_pde.item():.4e}"
            )
            if val_data is not None:
                with torch.no_grad():
                    x_val, u_val = val_data
                    val_mse = loss_fn(model(x_val), u_val)
                    msg += f" val_mse={val_mse.item():.4e}"
            print(msg)


if __name__ == "__main__":
    run()
