# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

import hydra
import numpy as np
import torch
from omegaconf import DictConfig as PhysicsNeMoConfig
from torch import nn
from torch.optim import Adam, lr_scheduler


class DGNet(nn.Module):
    def __init__(self, hidden_dim: int, num_layers: int):
        super().__init__()
        layers = [nn.Linear(2, hidden_dim), nn.Tanh()]
        for _ in range(max(0, num_layers - 1)):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, xy: torch.Tensor):
        return self.net(xy)


def grad(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y), create_graph=True)[0]


def poisson_residual(model: DGNet, xy: torch.Tensor):
    xy = xy.requires_grad_(True)
    u = model(xy)
    g = grad(u, xy)
    u_x, u_y = g[:, 0:1], g[:, 1:2]
    u_xx = grad(u_x, xy)[:, 0:1]
    u_yy = grad(u_y, xy)[:, 1:2]
    # -Delta u = 2
    return -(u_xx + u_yy) - 2.0


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def run(cfg: PhysicsNeMoConfig) -> None:
    torch.manual_seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DGNet(int(cfg.model.hidden_dim), int(cfg.model.num_layers)).to(device)
    optimizer = Adam(model.parameters(), lr=float(cfg.optimizer.lr))
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=float(cfg.scheduler.decay_rate))
    mse = nn.MSELoss()

    for step in range(int(cfg.training.max_steps)):
        optimizer.zero_grad()

        x = torch.empty(int(cfg.batch_size.interior), 1, device=device).uniform_(0.0, 1.0)
        y = torch.empty(int(cfg.batch_size.interior), 1, device=device).uniform_(0.0, 1.0)
        xy = torch.cat([x, y], dim=1)
        r = poisson_residual(model, xy)
        loss_interior = mse(r, torch.zeros_like(r))

        xb = torch.empty(int(cfg.batch_size.boundary), 1, device=device).uniform_(0.0, 1.0)
        yb = torch.empty(int(cfg.batch_size.boundary), 1, device=device).uniform_(0.0, 1.0)
        side = torch.randint(0, 4, (xb.shape[0],), device=device)
        xb[side == 0] = 0.0
        xb[side == 1] = 1.0
        yb[side == 2] = 0.0
        yb[side == 3] = 1.0
        xyb = torch.cat([xb, yb], dim=1)
        ub = model(xyb)
        g = ((xb - 1.0) ** 2) * (xb > 0.5).float() + (xb**2) * (xb <= 0.5).float()
        loss_boundary = mse(ub, g)

        xr = torch.empty(int(cfg.batch_size.rbf_functions), 1, device=device).uniform_(0.48, 0.52)
        yr = torch.empty(int(cfg.batch_size.rbf_functions), 1, device=device).uniform_(0.0, 1.0)
        ur = model(torch.cat([xr, yr], dim=1))
        loss_interface = mse(ur, torch.full_like(ur, 0.25))

        loss = (
            float(cfg.loss.interior) * loss_interior
            + float(cfg.loss.boundary) * loss_boundary
            + float(cfg.loss.interface) * loss_interface
        )

        loss.backward()
        optimizer.step()
        scheduler.step()

        if step % int(cfg.training.log_every) == 0:
            print(
                f"step={step:06d} total={loss.item():.4e} interior={loss_interior.item():.4e} "
                f"boundary={loss_boundary.item():.4e} interface={loss_interface.item():.4e}"
            )


if __name__ == "__main__":
    run()
