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
        layers = [nn.Linear(2, hidden_dim), nn.Tanh()]
        for _ in range(max(0, num_layers - 1)):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
        layers.append(nn.Linear(hidden_dim, 3))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        xy = torch.cat([x, y], dim=1)
        out = self.net(xy)
        return out[:, 0:1], out[:, 1:2], out[:, 2:3]


def grad(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y), create_graph=True)[0]


def sample_interior(n: int, x_bounds, y_bounds, radius: float, device: torch.device):
    out_x = []
    out_y = []
    target = n
    while len(out_x) < target:
        m = max(2 * (target - len(out_x)), 64)
        x = torch.empty(m, 1, device=device).uniform_(x_bounds[0], x_bounds[1])
        y = torch.empty(m, 1, device=device).uniform_(y_bounds[0], y_bounds[1])
        mask = (x.square() + y.square() >= radius * radius).squeeze(1)
        out_x.append(x[mask])
        out_y.append(y[mask])
        cur = sum(t.shape[0] for t in out_x)
        if cur >= target:
            break
    x = torch.cat(out_x, dim=0)[:target]
    y = torch.cat(out_y, dim=0)[:target]
    return x, y


def sample_cylinder(n: int, radius: float, device: torch.device):
    theta = torch.empty(n, 1, device=device).uniform_(0.0, 2.0 * np.pi)
    x = radius * torch.cos(theta)
    y = radius * torch.sin(theta)
    return x, y


def residuals(model: FlowNet, x: torch.Tensor, y: torch.Tensor, nu: float):
    x = x.requires_grad_(True)
    y = y.requires_grad_(True)
    u, v, p = model(x, y)

    u_x = grad(u, x)
    u_y = grad(u, y)
    v_x = grad(v, x)
    v_y = grad(v, y)
    p_x = grad(p, x)
    p_y = grad(p, y)
    u_xx = grad(u_x, x)
    u_yy = grad(u_y, y)
    v_xx = grad(v_x, x)
    v_yy = grad(v_y, y)

    continuity = u_x + v_y
    momentum_x = u * u_x + v * u_y + p_x - nu * (u_xx + u_yy)
    momentum_y = u * v_x + v * v_y + p_y - nu * (v_xx + v_yy)
    return continuity, momentum_x, momentum_y


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def run(cfg: PhysicsNeMoConfig) -> None:
    torch.manual_seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x_bounds = (float(cfg.geometry.x_min), float(cfg.geometry.x_max))
    y_bounds = (float(cfg.geometry.y_min), float(cfg.geometry.y_max))
    radius = float(cfg.geometry.cylinder_radius)
    nu = float(cfg.physics.nu)

    model = FlowNet(int(cfg.model.hidden_dim), int(cfg.model.num_layers)).to(device)
    optimizer = Adam(model.parameters(), lr=float(cfg.optimizer.lr))
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=float(cfg.scheduler.decay_rate))
    mse = nn.MSELoss()

    for step in range(int(cfg.training.max_steps)):
        optimizer.zero_grad()

        # Inlet: u=1, v=0
        y_in = torch.empty(int(cfg.batch_size.inlet), 1, device=device).uniform_(y_bounds[0], y_bounds[1])
        x_in = torch.full_like(y_in, x_bounds[0])
        u_in, v_in, _ = model(x_in, y_in)
        loss_in = mse(u_in, torch.ones_like(u_in)) + mse(v_in, torch.zeros_like(v_in))

        # Outlet: p=0
        y_out = torch.empty(int(cfg.batch_size.outlet), 1, device=device).uniform_(y_bounds[0], y_bounds[1])
        x_out = torch.full_like(y_out, x_bounds[1])
        _, _, p_out = model(x_out, y_out)
        loss_out = mse(p_out, torch.zeros_like(p_out))

        # Channel walls: u=1, v=0
        n_w = int(cfg.batch_size.walls)
        x_w = torch.empty(n_w, 1, device=device).uniform_(x_bounds[0], x_bounds[1])
        y_w = torch.where(
            torch.rand(n_w, 1, device=device) > 0.5,
            torch.full((n_w, 1), y_bounds[1], device=device),
            torch.full((n_w, 1), y_bounds[0], device=device),
        )
        u_w, v_w, _ = model(x_w, y_w)
        loss_w = mse(u_w, torch.ones_like(u_w)) + mse(v_w, torch.zeros_like(v_w))

        # Cylinder: no slip
        x_c, y_c = sample_cylinder(int(cfg.batch_size.no_slip), radius, device)
        u_c, v_c, _ = model(x_c, y_c)
        loss_c = mse(u_c, torch.zeros_like(u_c)) + mse(v_c, torch.zeros_like(v_c))

        # Interior PDE residual
        x_i, y_i = sample_interior(int(cfg.batch_size.interior), x_bounds, y_bounds, radius, device)
        cont, mx, my = residuals(model, x_i, y_i, nu)
        z = torch.zeros_like(cont)
        loss_pde = mse(cont, z) + mse(mx, z) + mse(my, z)

        loss = (
            float(cfg.loss.inlet) * loss_in
            + float(cfg.loss.outlet) * loss_out
            + float(cfg.loss.walls) * loss_w
            + float(cfg.loss.no_slip) * loss_c
            + float(cfg.loss.interior) * loss_pde
        )
        loss.backward()
        optimizer.step()
        scheduler.step()

        if step % int(cfg.training.log_every) == 0:
            print(
                f"step={step:06d} total={loss.item():.4e} inlet={loss_in.item():.4e} "
                f"outlet={loss_out.item():.4e} walls={loss_w.item():.4e} "
                f"no_slip={loss_c.item():.4e} interior={loss_pde.item():.4e}"
            )


if __name__ == "__main__":
    run()
