# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Surface PDE example adapted to an explicit PyTorch training loop.

This follows the original benchmark setup where we enforce two targets on sphere points:
- poisson_u = -18 * x * y * z
- flux_u = 0
"""

import hydra
import numpy as np
import torch
from omegaconf import DictConfig as PhysicsNeMoConfig
from torch import nn
from torch.optim import Adam, lr_scheduler


class SurfaceNet(nn.Module):
    def __init__(self, in_features: int, out_features: int, hidden_dim: int, num_layers: int):
        super().__init__()
        layers = [nn.Linear(in_features, hidden_dim), nn.Tanh()]
        for _ in range(max(0, num_layers - 1)):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
        layers.append(nn.Linear(hidden_dim, out_features))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def sample_sphere(n: int, device: torch.device) -> torch.Tensor:
    pts = torch.randn(n, 3, device=device)
    pts = pts / torch.linalg.norm(pts, dim=1, keepdim=True).clamp_min(1.0e-8)
    return pts


def poisson_and_flux(model: nn.Module, xyz: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    xyz = xyz.requires_grad_(True)
    u = model(xyz)
    grads = torch.autograd.grad(u, xyz, grad_outputs=torch.ones_like(u), create_graph=True)[0]

    u_x = grads[:, 0:1]
    u_y = grads[:, 1:2]
    u_z = grads[:, 2:3]

    u_xx = torch.autograd.grad(u_x, xyz, grad_outputs=torch.ones_like(u_x), create_graph=True)[0][:, 0:1]
    u_yy = torch.autograd.grad(u_y, xyz, grad_outputs=torch.ones_like(u_y), create_graph=True)[0][:, 1:2]
    u_zz = torch.autograd.grad(u_z, xyz, grad_outputs=torch.ones_like(u_z), create_graph=True)[0][:, 2:3]

    poisson_u = u_xx + u_yy + u_zz
    flux_u = (xyz[:, 0:1] * u_x) + (xyz[:, 1:2] * u_y) + (xyz[:, 2:3] * u_z)
    return poisson_u, flux_u


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def run(cfg: PhysicsNeMoConfig) -> None:
    torch.manual_seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SurfaceNet(
        in_features=3,
        out_features=1,
        hidden_dim=int(cfg.model.hidden_dim),
        num_layers=int(cfg.model.num_layers),
    ).to(device)

    optimizer = Adam(model.parameters(), lr=float(cfg.optimizer.lr))
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=float(cfg.scheduler.decay_rate))
    loss_fn = nn.MSELoss(reduction="mean")

    for step in range(int(cfg.training.max_steps)):
        optimizer.zero_grad()

        surface_pts = sample_sphere(int(cfg.batch_size.surface), device)
        poisson_u, flux_u = poisson_and_flux(model, surface_pts)

        target_poisson = -18.0 * (
            surface_pts[:, 0:1] * surface_pts[:, 1:2] * surface_pts[:, 2:3]
        )
        target_flux = torch.zeros_like(flux_u)

        loss_surface = loss_fn(poisson_u, target_poisson)
        loss_flux = loss_fn(flux_u, target_flux)

        point = torch.tensor([[1.0, 0.0, 0.0]], device=device)
        loss_point = loss_fn(model(point), torch.zeros((1, 1), device=device))

        loss = (
            float(cfg.loss.surface_poisson) * loss_surface
            + float(cfg.loss.surface_flux) * loss_flux
            + float(cfg.loss.point) * loss_point
        )
        loss.backward()
        optimizer.step()
        scheduler.step()

        if step % int(cfg.training.log_every) == 0:
            with torch.no_grad():
                val_pts = sample_sphere(int(cfg.batch_size.validation), device)
                pred_val = model(val_pts)
                true_val = val_pts[:, 0:1] * val_pts[:, 1:2] * val_pts[:, 2:3]
                val_mse = loss_fn(pred_val, true_val)
                print(
                    f"step={step:06d} total={loss.item():.4e} "
                    f"poisson={loss_surface.item():.4e} flux={loss_flux.item():.4e} "
                    f"point={loss_point.item():.4e} val_mse={val_mse.item():.4e}"
                )


if __name__ == "__main__":
    run()
