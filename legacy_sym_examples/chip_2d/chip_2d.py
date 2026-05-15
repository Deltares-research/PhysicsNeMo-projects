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
        out = self.net(torch.cat([x, y], dim=1))
        return out[:, 0:1], out[:, 1:2], out[:, 2:3]


def grad(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y), create_graph=True)[0]


def sample_interior(
    n: int,
    x_bounds,
    y_bounds,
    chip_pos: float,
    chip_w: float,
    chip_h: float,
    near_chip: bool,
    device: torch.device,
):
    xs, ys = [], []
    while sum(t.shape[0] for t in xs) < n:
        m = max(2 * (n - sum(t.shape[0] for t in xs)), 256)
        x = torch.empty(m, 1, device=device).uniform_(x_bounds[0], x_bounds[1])
        y = torch.empty(m, 1, device=device).uniform_(y_bounds[0], y_bounds[1])
        in_chip = (x >= chip_pos) & (x <= chip_pos + chip_w) & (y >= y_bounds[0]) & (y <= y_bounds[0] + chip_h)
        if near_chip:
            slab = (x >= chip_pos - 0.25) & (x <= chip_pos + chip_w + 0.25)
        else:
            slab = (x < chip_pos - 0.25) | (x > chip_pos + chip_w + 0.25)
        mask = (~in_chip & slab).squeeze(1)
        xs.append(x[mask])
        ys.append(y[mask])
    return torch.cat(xs, dim=0)[:n], torch.cat(ys, dim=0)[:n]


def sample_no_slip(n: int, x_bounds, y_bounds, chip_pos: float, chip_w: float, chip_h: float, device: torch.device):
    n_wall = max(1, n // 2)
    n_chip = n - n_wall

    # channel walls
    xw = torch.empty(n_wall, 1, device=device).uniform_(x_bounds[0], x_bounds[1])
    yw = torch.where(
        torch.rand(n_wall, 1, device=device) > 0.5,
        torch.full((n_wall, 1), y_bounds[1], device=device),
        torch.full((n_wall, 1), y_bounds[0], device=device),
    )

    # chip top and side faces
    nc_face = max(1, n_chip // 3)
    xt = torch.empty(nc_face, 1, device=device).uniform_(chip_pos, chip_pos + chip_w)
    yt = torch.full((nc_face, 1), y_bounds[0] + chip_h, device=device)

    yl = torch.empty(nc_face, 1, device=device).uniform_(y_bounds[0], y_bounds[0] + chip_h)
    xl = torch.full((nc_face, 1), chip_pos, device=device)

    yr = torch.empty(n_chip - 2 * nc_face, 1, device=device).uniform_(y_bounds[0], y_bounds[0] + chip_h)
    xr = torch.full((n_chip - 2 * nc_face, 1), chip_pos + chip_w, device=device)

    x = torch.cat([xw, xt, xl, xr], dim=0)
    y = torch.cat([yw, yt, yl, yr], dim=0)
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


def inlet_profile(y: torch.Tensor, y0: float, y1: float, height: float):
    yc = 0.5 * (y0 + y1)
    r = 0.5 * (y1 - y0)
    yn = (y - yc) / max(r, 1.0e-8)
    return height * (1.0 - yn.square())


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def run(cfg: PhysicsNeMoConfig) -> None:
    torch.manual_seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x_bounds = (float(cfg.geometry.x_min), float(cfg.geometry.x_max))
    y_bounds = (float(cfg.geometry.y_min), float(cfg.geometry.y_max))
    chip_pos = float(cfg.geometry.chip_pos)
    chip_w = float(cfg.geometry.chip_width)
    chip_h = float(cfg.geometry.chip_height)
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
        vf["x"] -= 2.5
        vf["y"] -= 0.5
        val_data = {
            "x": torch.as_tensor(vf["x"], dtype=torch.float32, device=device),
            "y": torch.as_tensor(vf["y"], dtype=torch.float32, device=device),
            "u": torch.as_tensor(vf["u"], dtype=torch.float32, device=device),
            "v": torch.as_tensor(vf["v"], dtype=torch.float32, device=device),
            "p": torch.as_tensor(vf["p"], dtype=torch.float32, device=device),
        }

    for step in range(int(cfg.training.max_steps)):
        optimizer.zero_grad()

        y_in = torch.empty(int(cfg.batch_size.inlet), 1, device=device).uniform_(y_bounds[0], y_bounds[1])
        x_in = torch.full_like(y_in, x_bounds[0])
        u_in, v_in, _ = model(x_in, y_in)
        loss_inlet = mse(u_in, inlet_profile(y_in, y_bounds[0], y_bounds[1], inlet_vel)) + mse(v_in, torch.zeros_like(v_in))

        y_out = torch.empty(int(cfg.batch_size.outlet), 1, device=device).uniform_(y_bounds[0], y_bounds[1])
        x_out = torch.full_like(y_out, x_bounds[1])
        _, _, p_out = model(x_out, y_out)
        loss_outlet = mse(p_out, torch.zeros_like(p_out))

        x_ns, y_ns = sample_no_slip(int(cfg.batch_size.no_slip), x_bounds, y_bounds, chip_pos, chip_w, chip_h, device)
        u_ns, v_ns, _ = model(x_ns, y_ns)
        loss_ns = mse(u_ns, torch.zeros_like(u_ns)) + mse(v_ns, torch.zeros_like(v_ns))

        x_lr, y_lr = sample_interior(
            int(cfg.batch_size.interior_lr),
            x_bounds,
            y_bounds,
            chip_pos,
            chip_w,
            chip_h,
            near_chip=False,
            device=device,
        )
        x_hr, y_hr = sample_interior(
            int(cfg.batch_size.interior_hr),
            x_bounds,
            y_bounds,
            chip_pos,
            chip_w,
            chip_h,
            near_chip=True,
            device=device,
        )
        cont_lr, mx_lr, my_lr = residuals(model, x_lr, y_lr, nu)
        cont_hr, mx_hr, my_hr = residuals(model, x_hr, y_hr, nu)
        z_lr = torch.zeros_like(cont_lr)
        z_hr = torch.zeros_like(cont_hr)
        loss_pde = (
            mse(cont_lr, z_lr)
            + mse(mx_lr, z_lr)
            + mse(my_lr, z_lr)
            + mse(cont_hr, z_hr)
            + mse(mx_hr, z_hr)
            + mse(my_hr, z_hr)
        )

        # integral continuity approximated by random outlet slices
        flux_losses = []
        for _ in range(int(cfg.batch_size.num_integral_continuity)):
            y_flux = torch.empty(int(cfg.batch_size.integral_continuity), 1, device=device).uniform_(y_bounds[0], y_bounds[1])
            x_flux = torch.full_like(y_flux, torch.empty(1, device=device).uniform_(x_bounds[0], x_bounds[1]).item())
            u_flux, _, _ = model(x_flux, y_flux)
            flux_losses.append((y_bounds[1] - y_bounds[0]) * torch.mean(u_flux))
        flux_est = torch.stack(flux_losses).mean()
        loss_flux = mse(flux_est, torch.tensor(1.0, device=device))

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
