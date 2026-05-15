# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

import hydra
import numpy as np
import torch
from omegaconf import DictConfig as PhysicsNeMoConfig
from torch import nn
from torch.optim import Adam, lr_scheduler


class TurbNet(nn.Module):
    def __init__(self, hidden_dim: int, num_layers: int):
        super().__init__()
        layers = [nn.Linear(2, hidden_dim), nn.Tanh()]
        for _ in range(max(0, num_layers - 1)):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
        layers.append(nn.Linear(hidden_dim, 5))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        out = self.net(torch.cat([x, y], dim=1))
        u, v, p = out[:, 0:1], out[:, 1:2], out[:, 2:3]
        k = torch.nn.functional.softplus(out[:, 3:4])
        ep = torch.nn.functional.softplus(out[:, 4:5])
        return u, v, p, k, ep


def grad(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y), create_graph=True)[0]


def residuals(model: TurbNet, x: torch.Tensor, y: torch.Tensor, nu: float):
    x = x.requires_grad_(True)
    y = y.requires_grad_(True)
    u, v, p, _, _ = model(x, y)

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


@hydra.main(version_base="1.3", config_path="conf_re590_k_ep_LS", config_name="config")
def run(cfg: PhysicsNeMoConfig) -> None:
    torch.manual_seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    re = float(cfg.physics.Re)
    nu = 1.0 / re
    y_plus = float(cfg.physics.y_plus)
    resolved_y_start = y_plus * nu
    x_bounds = (-np.pi / 2, np.pi / 2)
    y_res = (-1.0 + resolved_y_start, 1.0 - resolved_y_start)

    model = TurbNet(int(cfg.model.hidden_dim), int(cfg.model.num_layers)).to(device)
    optimizer = Adam(model.parameters(), lr=float(cfg.optimizer.lr))
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=float(cfg.scheduler.decay_rate))
    mse = nn.MSELoss()

    p_grad = float(cfg.physics.p_grad)
    p_in = p_grad * (x_bounds[1] - x_bounds[0])

    for step in range(int(cfg.training.max_steps)):
        optimizer.zero_grad()

        x_wf = torch.empty(int(cfg.batch_size.wf_pt), 1, device=device).uniform_(x_bounds[0], x_bounds[1])
        y_wf = torch.where(
            torch.rand_like(x_wf) > 0.5,
            torch.full_like(x_wf, y_res[0]),
            torch.full_like(x_wf, y_res[1]),
        )
        u_wf, v_wf, _, _, ep_wf = model(x_wf, y_wf)
        loss_wf = mse(u_wf, torch.zeros_like(u_wf)) + mse(v_wf, torch.zeros_like(v_wf)) + mse(ep_wf, torch.zeros_like(ep_wf))

        x_i = torch.empty(int(cfg.batch_size.interior), 1, device=device).uniform_(x_bounds[0], x_bounds[1])
        y_i = torch.empty(int(cfg.batch_size.interior), 1, device=device).uniform_(y_res[0], y_res[1])
        cont, mx, my = residuals(model, x_i, y_i, nu)
        z = torch.zeros_like(cont)
        loss_pde = mse(cont, z) + mse(mx, z) + mse(my, z)

        y_in = torch.empty(int(cfg.batch_size.inlet), 1, device=device).uniform_(y_res[0], y_res[1])
        x_in = torch.full_like(y_in, x_bounds[0])
        _, _, p_i, _, _ = model(x_in, y_in)
        loss_in = mse(p_i, torch.full_like(p_i, p_in))

        y_out = torch.empty(int(cfg.batch_size.outlet), 1, device=device).uniform_(y_res[0], y_res[1])
        x_out = torch.full_like(y_out, x_bounds[1])
        _, _, p_o, _, _ = model(x_out, y_out)
        loss_out = mse(p_o, torch.zeros_like(p_o))

        x0 = torch.empty(int(cfg.batch_size.interior_init), 1, device=device).uniform_(x_bounds[0], x_bounds[1])
        y0 = torch.empty(int(cfg.batch_size.interior_init), 1, device=device).uniform_(y_res[0], y_res[1])
        u0, v0, p0, k0, ep0 = model(x0, y0)
        loss_init = mse(u0, torch.zeros_like(u0)) + mse(v0, torch.zeros_like(v0)) + mse(p0, torch.zeros_like(p0)) + mse(k0, torch.zeros_like(k0)) + mse(ep0, torch.zeros_like(ep0))

        loss = (
            float(cfg.loss.wf_pt) * loss_wf
            + float(cfg.loss.interior) * loss_pde
            + float(cfg.loss.inlet) * loss_in
            + float(cfg.loss.outlet) * loss_out
            + float(cfg.loss.interior_init) * loss_init
        )

        loss.backward()
        optimizer.step()
        scheduler.step()

        if step % int(cfg.training.log_every) == 0:
            print(
                f"step={step:06d} total={loss.item():.4e} wf={loss_wf.item():.4e} "
                f"pde={loss_pde.item():.4e} inlet={loss_in.item():.4e} "
                f"outlet={loss_out.item():.4e} init={loss_init.item():.4e}"
            )


if __name__ == "__main__":
    run()
