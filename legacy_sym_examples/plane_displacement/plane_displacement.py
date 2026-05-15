# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

import hydra
import numpy as np
import torch
from omegaconf import DictConfig as PhysicsNeMoConfig
from torch import nn
from torch.optim import Adam, lr_scheduler


class ElasticityNet(nn.Module):
    def __init__(self, hidden_dim: int, num_layers: int):
        super().__init__()
        layers = [nn.Linear(2, hidden_dim), nn.Tanh()]
        for _ in range(max(0, num_layers - 1)):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
        layers.append(nn.Linear(hidden_dim, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        uv = self.net(torch.cat([x, y], dim=1))
        return uv[:, 0:1], uv[:, 1:2]


def grad(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y), create_graph=True)[0]


def elasticity_residual(model: ElasticityNet, x: torch.Tensor, y: torch.Tensor, lame_l: float, lame_m: float):
    x = x.requires_grad_(True)
    y = y.requires_grad_(True)
    u, v = model(x, y)

    u_x, u_y = grad(u, x), grad(u, y)
    v_x, v_y = grad(v, x), grad(v, y)

    w_z = -lame_l / (lame_l + 2.0 * lame_m) * (u_x + v_y)
    sigma_xx = lame_l * (u_x + v_y + w_z) + 2.0 * lame_m * u_x
    sigma_yy = lame_l * (u_x + v_y + w_z) + 2.0 * lame_m * v_y
    sigma_xy = lame_m * (u_y + v_x)

    sigma_xx_x = grad(sigma_xx, x)
    sigma_xy_y = grad(sigma_xy, y)
    sigma_xy_x = grad(sigma_xy, x)
    sigma_yy_y = grad(sigma_yy, y)

    eq_x = sigma_xx_x + sigma_xy_y
    eq_y = sigma_xy_x + sigma_yy_y
    return eq_x, eq_y


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def run(cfg: PhysicsNeMoConfig) -> None:
    torch.manual_seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    domain_origin = (-0.5, -0.5)
    domain_dim = (1.0, 1.0)
    x_bounds = (domain_origin[0], domain_origin[0] + domain_dim[0])
    y_bounds = (domain_origin[1], domain_origin[1] + domain_dim[1])

    e_mod = float(cfg.physics.E)
    nu = float(cfg.physics.nu)
    lame_l = nu * e_mod / ((1.0 + nu) * (1.0 - 2.0 * nu))
    lame_m = e_mod / (2.0 * (1.0 + nu))

    model = ElasticityNet(int(cfg.model.hidden_dim), int(cfg.model.num_layers)).to(device)
    optimizer = Adam(model.parameters(), lr=float(cfg.optimizer.lr))
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=float(cfg.scheduler.decay_rate))
    mse = nn.MSELoss()

    top_x_max = domain_origin[0] + domain_dim[0] / 2.0

    for step in range(int(cfg.training.max_steps)):
        optimizer.zero_grad()

        # Bottom boundary clamped.
        xb = torch.empty(int(cfg.batch_size.bottom), 1, device=device).uniform_(x_bounds[0], x_bounds[1])
        yb = torch.full_like(xb, y_bounds[0])
        ub, vb = model(xb, yb)
        loss_bottom = mse(ub, torch.zeros_like(ub)) + mse(vb, torch.zeros_like(vb))

        # Top boundary displacement on left half.
        xt = torch.empty(int(cfg.batch_size.top), 1, device=device).uniform_(x_bounds[0], top_x_max)
        yt = torch.full_like(xt, y_bounds[1])
        ut, vt = model(xt, yt)
        loss_top = mse(ut, torch.zeros_like(ut)) + mse(vt, torch.full_like(vt, 0.1))

        # Interior elasticity residual.
        xi = torch.empty(int(cfg.batch_size.interior), 1, device=device).uniform_(x_bounds[0], x_bounds[1])
        yi = torch.empty(int(cfg.batch_size.interior), 1, device=device).uniform_(y_bounds[0], y_bounds[1])
        ex, ey = elasticity_residual(model, xi, yi, lame_l, lame_m)
        z = torch.zeros_like(ex)
        loss_interior = mse(ex, z) + mse(ey, z)

        loss = (
            float(cfg.loss.bottom) * loss_bottom
            + float(cfg.loss.top) * loss_top
            + float(cfg.loss.interior) * loss_interior
        )
        loss.backward()
        optimizer.step()
        scheduler.step()

        if step % int(cfg.training.log_every) == 0:
            print(
                f"step={step:06d} total={loss.item():.4e} bottom={loss_bottom.item():.4e} "
                f"top={loss_top.item():.4e} interior={loss_interior.item():.4e}"
            )


if __name__ == "__main__":
    run()
