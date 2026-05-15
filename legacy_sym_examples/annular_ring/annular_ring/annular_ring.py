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


def sample_interior(n: int, x_bounds, y_bounds, r_outer: float, r_inner: float, device: torch.device):
    xs = []
    ys = []
    while sum(t.shape[0] for t in xs) < n:
        m = max(2 * (n - sum(t.shape[0] for t in xs)), 256)
        x = torch.empty(m, 1, device=device).uniform_(x_bounds[0], x_bounds[1])
        y = torch.empty(m, 1, device=device).uniform_(y_bounds[0], y_bounds[1])
        r2 = x.square() + y.square()
        in_rect = torch.abs(y) <= 1.0
        in_outer = r2 <= r_outer * r_outer
        out_inner = r2 >= r_inner * r_inner
        mask = ((in_rect | in_outer) & out_inner).squeeze(1)
        xs.append(x[mask])
        ys.append(y[mask])
    return torch.cat(xs, dim=0)[:n], torch.cat(ys, dim=0)[:n]


def sample_no_slip(n: int, x_bounds, r_outer: float, r_inner: float, device: torch.device):
    n_each = max(1, n // 4)
    # channel walls
    xw = torch.empty(2 * n_each, 1, device=device).uniform_(x_bounds[0], x_bounds[1])
    yw = torch.cat(
        [torch.full((n_each, 1), -1.0, device=device), torch.full((n_each, 1), 1.0, device=device)],
        dim=0,
    )

    # outer and inner circle
    th_o = torch.empty(n_each, 1, device=device).uniform_(0.0, 2.0 * np.pi)
    xo = r_outer * torch.cos(th_o)
    yo = r_outer * torch.sin(th_o)

    th_i = torch.empty(n_each, 1, device=device).uniform_(0.0, 2.0 * np.pi)
    xi = r_inner * torch.cos(th_i)
    yi = r_inner * torch.sin(th_i)

    x = torch.cat([xw, xo, xi], dim=0)
    y = torch.cat([yw, yo, yi], dim=0)
    return x[:n], y[:n]


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


def inlet_profile(y: torch.Tensor, height: float):
    return height * (1.0 - y.square())


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def run(cfg: PhysicsNeMoConfig) -> None:
    torch.manual_seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x_bounds = (float(cfg.geometry.x_min), float(cfg.geometry.x_max))
    y_bounds = (float(cfg.geometry.y_min), float(cfg.geometry.y_max))
    r_outer = float(cfg.geometry.outer_radius)
    r_inner = float(cfg.geometry.inner_radius)
    inlet_vel = float(cfg.physics.inlet_vel)
    nu = float(cfg.physics.nu)

    model = FlowNet(int(cfg.model.hidden_dim), int(cfg.model.num_layers)).to(device)
    optimizer = Adam(model.parameters(), lr=float(cfg.optimizer.lr))
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=float(cfg.scheduler.decay_rate))
    mse = nn.MSELoss()

    val_csv = to_absolute_path(str(cfg.data.validation_csv))
    val_data = None
    if os.path.exists(val_csv):
        mapping = {"Points:0": "x", "Points:1": "y", "U:0": "u", "U:1": "v", "p": "p"}
        vf = csv_to_dict(val_csv, mapping)
        vf["x"] += x_bounds[0]
        val_data = {
            "x": torch.as_tensor(vf["x"], dtype=torch.float32, device=device),
            "y": torch.as_tensor(vf["y"], dtype=torch.float32, device=device),
            "u": torch.as_tensor(vf["u"], dtype=torch.float32, device=device),
            "v": torch.as_tensor(vf["v"], dtype=torch.float32, device=device),
            "p": torch.as_tensor(vf["p"], dtype=torch.float32, device=device),
        }

    for step in range(int(cfg.training.max_steps)):
        optimizer.zero_grad()

        # inlet
        y_in = torch.empty(int(cfg.batch_size.inlet), 1, device=device).uniform_(-1.0, 1.0)
        x_in = torch.full_like(y_in, x_bounds[0])
        u_in, v_in, _ = model(x_in, y_in)
        loss_inlet = mse(u_in, inlet_profile(y_in, inlet_vel)) + mse(v_in, torch.zeros_like(v_in))

        # outlet
        y_out = torch.empty(int(cfg.batch_size.outlet), 1, device=device).uniform_(-1.0, 1.0)
        x_out = torch.full_like(y_out, x_bounds[1])
        _, _, p_out = model(x_out, y_out)
        loss_outlet = mse(p_out, torch.zeros_like(p_out))

        # no-slip
        x_ns, y_ns = sample_no_slip(int(cfg.batch_size.no_slip), x_bounds, r_outer, r_inner, device)
        u_ns, v_ns, _ = model(x_ns, y_ns)
        loss_ns = mse(u_ns, torch.zeros_like(u_ns)) + mse(v_ns, torch.zeros_like(v_ns))

        # interior PDE
        x_i, y_i = sample_interior(int(cfg.batch_size.interior), x_bounds, y_bounds, r_outer, r_inner, device)
        cont, mx, my = residuals(model, x_i, y_i, nu)
        z = torch.zeros_like(cont)
        loss_pde = mse(cont, z) + mse(mx, z) + mse(my, z)

        # integral continuity at outlet: integral u dy = 2.0 over y in [-1,1]
        u_flux, _, _ = model(x_out, y_out)
        flux_est = 2.0 * torch.mean(u_flux)
        loss_flux = mse(flux_est, torch.tensor(2.0, device=device))

        loss = (
            float(cfg.loss.inlet) * loss_inlet
            + float(cfg.loss.outlet) * loss_outlet
            + float(cfg.loss.no_slip) * loss_ns
            + float(cfg.loss.interior) * loss_pde
            + float(cfg.loss.integral_continuity) * loss_flux
        )
        loss.backward()
        optimizer.step()
        scheduler.step()

        if step % int(cfg.training.log_every) == 0:
            msg = (
                f"step={step:06d} total={loss.item():.4e} inlet={loss_inlet.item():.4e} "
                f"outlet={loss_outlet.item():.4e} no_slip={loss_ns.item():.4e} "
                f"interior={loss_pde.item():.4e} flux={loss_flux.item():.4e}"
            )
            if val_data is not None:
                with torch.no_grad():
                    u_v, v_v, p_v = model(val_data["x"], val_data["y"])
                    val_loss = mse(u_v, val_data["u"]) + mse(v_v, val_data["v"]) + mse(p_v, val_data["p"])
                    msg += f" val={val_loss.item():.4e}"
            print(msg)


if __name__ == "__main__":
    run()
