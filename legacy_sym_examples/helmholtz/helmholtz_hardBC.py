# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
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


def hard_bc_ansatz(u_star: torch.Tensor, coords: torch.Tensor, width: float, height: float):
    x = coords[:, 0:1]
    y = coords[:, 1:2]
    phi_x = (x + width / 2.0) * (width / 2.0 - x)
    phi_y = (y + height / 2.0) * (height / 2.0 - y)
    return phi_x * phi_y * u_star


def pde_residual(model: nn.Module, coords: torch.Tensor, width: float, height: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    coords = coords.requires_grad_(True)
    u_star = model(coords)
    u = hard_bc_ansatz(u_star, coords, width, height)

    grad_u = torch.autograd.grad(u, coords, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_x = grad_u[:, 0:1]
    u_y = grad_u[:, 1:2]
    u_xx = torch.autograd.grad(u_x, coords, grad_outputs=torch.ones_like(u_x), create_graph=True)[0][:, 0:1]
    u_yy = torch.autograd.grad(u_y, coords, grad_outputs=torch.ones_like(u_y), create_graph=True)[0][:, 1:2]

    x = coords[:, 0:1]
    y = coords[:, 1:2]
    f = forcing_term(x, y)
    helmholtz = u_xx + u_yy + u - f

    grad_u_star = torch.autograd.grad(u_star, coords, grad_outputs=torch.ones_like(u_star), create_graph=True)[0]
    u_star_x = grad_u_star[:, 0:1]
    u_star_y = grad_u_star[:, 1:2]
    return helmholtz, u_star_x, u_star_y


def sample_interior(n: int, width: float, height: float, device: torch.device) -> torch.Tensor:
    x = (torch.rand(n, 1, device=device) - 0.5) * width
    y = (torch.rand(n, 1, device=device) - 0.5) * height
    return torch.cat([x, y], dim=1)


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


@hydra.main(version_base="1.3", config_path="conf", config_name="config_hardBC")
def run(cfg: PhysicsNeMoConfig) -> None:
    torch.manual_seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    width = float(cfg.domain.width)
    height = float(cfg.domain.height)

    model = HelmholtzNet(2, 1, int(cfg.model.hidden_dim), int(cfg.model.num_layers)).to(device)
    mse = nn.MSELoss(reduction="mean")
    optimizer = Adam(model.parameters(), lr=float(cfg.optimizer.lr))
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=float(cfg.scheduler.decay_rate))

    val_data = maybe_load_validation(device)

    for step in range(int(cfg.training.max_steps)):
        optimizer.zero_grad()
        interior = sample_interior(int(cfg.batch_size.interior), width, height, device)

        helmholtz, ux, uy = pde_residual(model, interior, width, height)
        loss_h = mse(helmholtz, torch.zeros_like(helmholtz))
        loss_cx = mse(ux, torch.zeros_like(ux))
        loss_cy = mse(uy, torch.zeros_like(uy))
        loss = (
            float(cfg.loss.helmholtz) * loss_h
            + float(cfg.loss.compatibility_u_x) * loss_cx
            + float(cfg.loss.compatibility_u_y) * loss_cy
        )

        loss.backward()
        optimizer.step()
        scheduler.step()

        if step % int(cfg.training.log_every) == 0:
            msg = (
                f"step={step:06d} total={loss.item():.4e} helmholtz={loss_h.item():.4e} "
                f"compat_x={loss_cx.item():.4e} compat_y={loss_cy.item():.4e}"
            )
            if val_data is not None:
                with torch.no_grad():
                    x_val, u_val = val_data
                    u_pred = hard_bc_ansatz(model(x_val), x_val, width, height)
                    val_mse = mse(u_pred, u_val)
                    msg += f" val_mse={val_mse.item():.4e}"
            print(msg)


if __name__ == "__main__":
    run()
