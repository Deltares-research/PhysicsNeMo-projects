# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

import os

import hydra
import numpy as np
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig as PhysicsNeMoConfig
from torch import nn
from torch.optim import Adam, lr_scheduler


def csv_to_dict(path, mapping=None):
    data = np.genfromtxt(path, delimiter=",", names=True)
    if data.ndim == 0:
        data = np.array([data], dtype=data.dtype)
    out = {}
    for name in data.dtype.names:
        key = mapping.get(name, name) if mapping else name
        out[key] = np.asarray(data[name], dtype=np.float32)[:, None]
    return out


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int, num_layers: int):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden_dim), nn.Tanh()]
        for _ in range(max(0, num_layers - 1)):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class InverseModel(nn.Module):
    def __init__(self, hidden_dim: int, num_layers: int):
        super().__init__()
        self.state_net = MLP(2, 4, hidden_dim, num_layers)
        self.log_nu = nn.Parameter(torch.tensor(-6.0))
        self.log_D = nn.Parameter(torch.tensor(-6.0))

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        xy = torch.cat([x, y], dim=1)
        out = self.state_net(xy)
        u, v, p, c = out[:, 0:1], out[:, 1:2], out[:, 2:3], out[:, 3:4]
        nu = torch.nn.functional.softplus(self.log_nu) + 1.0e-8
        D = torch.nn.functional.softplus(self.log_D) + 1.0e-8
        return u, v, p, c, nu, D


def grad(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y), create_graph=True)[0]


def residuals(model: InverseModel, x: torch.Tensor, y: torch.Tensor):
    x = x.requires_grad_(True)
    y = y.requires_grad_(True)
    u, v, p, c, nu, D = model(x, y)

    u_x = grad(u, x)
    u_y = grad(u, y)
    v_x = grad(v, x)
    v_y = grad(v, y)
    p_x = grad(p, x)
    p_y = grad(p, y)
    c_x = grad(c, x)
    c_y = grad(c, y)

    u_xx = grad(u_x, x)
    u_yy = grad(u_y, y)
    v_xx = grad(v_x, x)
    v_yy = grad(v_y, y)
    c_xx = grad(c_x, x)
    c_yy = grad(c_y, y)

    continuity = u_x + v_y
    momentum_x = u * u_x + v * u_y + p_x - nu * (u_xx + u_yy)
    momentum_y = u * v_x + v * v_y + p_y - nu * (v_xx + v_yy)
    advection_diffusion_c = u * c_x + v * c_y - D * (c_xx + c_yy)

    return continuity, momentum_x, momentum_y, advection_diffusion_c, nu, D


@hydra.main(version_base="1.3", config_path="conf_inverse", config_name="config")
def run(cfg: PhysicsNeMoConfig) -> None:
    torch.manual_seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_temp = 293.498
    file_path = to_absolute_path(str(cfg.data.file_path))
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Missing data file: {file_path}. Download supplemental materials and place the OpenFOAM CSV at this path."
        )

    mapping = {
        "Points:0": "x",
        "Points:1": "y",
        "U:0": "u",
        "U:1": "v",
        "p": "p",
        "T": "c",
    }
    openfoam_var = csv_to_dict(file_path, mapping)
    openfoam_var["c"] = openfoam_var["c"] / base_temp - 1.0

    x_np = openfoam_var["x"]
    y_np = openfoam_var["y"]
    u_np = openfoam_var["u"]
    v_np = openfoam_var["v"]
    p_np = openfoam_var["p"]
    c_np = openfoam_var["c"]

    x_all = torch.as_tensor(x_np, dtype=torch.float32, device=device)
    y_all = torch.as_tensor(y_np, dtype=torch.float32, device=device)
    target = torch.as_tensor(np.concatenate([u_np, v_np, p_np, c_np], axis=1), dtype=torch.float32, device=device)

    model = InverseModel(int(cfg.model.hidden_dim), int(cfg.model.num_layers)).to(device)
    optimizer = Adam(model.parameters(), lr=float(cfg.optimizer.lr))
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=float(cfg.scheduler.decay_rate))
    mse = nn.MSELoss()

    n_total = x_all.shape[0]
    batch_size = min(int(cfg.batch_size.data), n_total)

    for step in range(int(cfg.training.max_steps)):
        idx = torch.randint(0, n_total, (batch_size,), device=device)
        xb = x_all[idx]
        yb = y_all[idx]
        tb = target[idx]

        optimizer.zero_grad()

        u, v, p, c, _, _ = model(xb, yb)
        pred = torch.cat([u, v, p, c], dim=1)
        loss_data = mse(pred, tb)

        cont, mx, my, ad, nu, D = residuals(model, xb, yb)
        zero = torch.zeros_like(cont)
        loss_pde = mse(cont, zero) + mse(mx, zero) + mse(my, zero) + mse(ad, zero)

        loss = float(cfg.loss.data_weight) * loss_data + float(cfg.loss.pde_weight) * loss_pde
        loss.backward()
        optimizer.step()
        scheduler.step()

        if step % int(cfg.training.log_every) == 0:
            print(
                f"step={step:06d} total={loss.item():.4e} data={loss_data.item():.4e} "
                f"pde={loss_pde.item():.4e} nu={nu.item():.4e} D={D.item():.4e}"
            )


if __name__ == "__main__":
    run()
