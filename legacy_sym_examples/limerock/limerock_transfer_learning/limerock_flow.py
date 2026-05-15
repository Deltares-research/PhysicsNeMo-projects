# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

import hydra
import numpy as np
import torch
import sys
from pathlib import Path
from omegaconf import DictConfig as PhysicsNeMoConfig
from torch import nn
from torch.optim import Adam, lr_scheduler

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from limerock_properties import inlet_velocity_normalized, limerock, nu, volumetric_flow


class FlowNet(nn.Module):
    def __init__(self, hidden_dim: int, num_layers: int):
        super().__init__()
        layers = [nn.Linear(3, hidden_dim), nn.Tanh()]
        for _ in range(max(0, num_layers - 1)):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
        layers.append(nn.Linear(hidden_dim, 4))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor):
        out = self.net(torch.cat([x, y, z], dim=1))
        return out[:, 0:1], out[:, 1:2], out[:, 2:3], out[:, 3:4]


def grad(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y), create_graph=True)[0]


def residuals(model: FlowNet, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, nu_val: float):
    x = x.requires_grad_(True)
    y = y.requires_grad_(True)
    z = z.requires_grad_(True)

    u, v, w, p = model(x, y, z)

    u_x, u_y, u_z = grad(u, x), grad(u, y), grad(u, z)
    v_x, v_y, v_z = grad(v, x), grad(v, y), grad(v, z)
    w_x, w_y, w_z = grad(w, x), grad(w, y), grad(w, z)
    p_x, p_y, p_z = grad(p, x), grad(p, y), grad(p, z)

    u_xx, u_yy, u_zz = grad(u_x, x), grad(u_y, y), grad(u_z, z)
    v_xx, v_yy, v_zz = grad(v_x, x), grad(v_y, y), grad(v_z, z)
    w_xx, w_yy, w_zz = grad(w_x, x), grad(w_y, y), grad(w_z, z)

    continuity = u_x + v_y + w_z
    momentum_x = u * u_x + v * u_y + w * u_z + p_x - nu_val * (u_xx + u_yy + u_zz)
    momentum_y = u * v_x + v * v_y + w * v_z + p_y - nu_val * (v_xx + v_yy + v_zz)
    momentum_z = u * w_x + v * w_y + w * w_z + p_z - nu_val * (w_xx + w_yy + w_zz)
    return continuity, momentum_x, momentum_y, momentum_z


def to_tensors(sample_dict, device):
    return (
        torch.as_tensor(sample_dict["x"], dtype=torch.float32, device=device),
        torch.as_tensor(sample_dict["y"], dtype=torch.float32, device=device),
        torch.as_tensor(sample_dict["z"], dtype=torch.float32, device=device),
    )


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def run(cfg: PhysicsNeMoConfig) -> None:
    torch.manual_seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = FlowNet(int(cfg.model.hidden_dim), int(cfg.model.num_layers)).to(device)
    optimizer = Adam(model.parameters(), lr=float(cfg.optimizer.lr))
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=float(cfg.scheduler.decay_rate))
    mse = nn.MSELoss()

    nu_val = float(nu)

    for step in range(int(cfg.training.max_steps)):
        optimizer.zero_grad()

        xi, yi, zi = to_tensors(limerock.inlet.sample_boundary(int(cfg.batch_size.inlet)), device)
        u_i, v_i, w_i, _ = model(xi, yi, zi)
        loss_in = mse(u_i, torch.full_like(u_i, float(inlet_velocity_normalized))) + mse(v_i, torch.zeros_like(v_i)) + mse(w_i, torch.zeros_like(w_i))

        xo, yo, zo = to_tensors(limerock.outlet.sample_boundary(int(cfg.batch_size.outlet)), device)
        _, _, _, p_o = model(xo, yo, zo)
        loss_out = mse(p_o, torch.zeros_like(p_o))

        xn, yn, zn = to_tensors(limerock.geo.sample_boundary(int(cfg.batch_size.no_slip)), device)
        u_n, v_n, w_n, _ = model(xn, yn, zn)
        loss_ns = mse(u_n, torch.zeros_like(u_n)) + mse(v_n, torch.zeros_like(v_n)) + mse(w_n, torch.zeros_like(w_n))

        xl, yl, zl = to_tensors(limerock.geo.sample_interior(int(cfg.batch_size.lr_interior), bounds=limerock.geo_bounds), device)
        xh, yh, zh = to_tensors(limerock.geo.sample_interior(int(cfg.batch_size.hr_interior), bounds=limerock.geo_hr_bounds), device)

        c_l, mx_l, my_l, mz_l = residuals(model, xl, yl, zl, nu_val)
        c_h, mx_h, my_h, mz_h = residuals(model, xh, yh, zh, nu_val)
        z_l = torch.zeros_like(c_l)
        z_h = torch.zeros_like(c_h)
        loss_pde = (
            mse(c_l, z_l) + mse(mx_l, z_l) + mse(my_l, z_l) + mse(mz_l, z_l)
            + mse(c_h, z_h) + mse(mx_h, z_h) + mse(my_h, z_h) + mse(mz_h, z_h)
        )

        xf, yf, zf = to_tensors(limerock.outlet.sample_boundary(int(cfg.batch_size.integral_continuity)), device)
        u_f, _, _, _ = model(xf, yf, zf)
        flux_est = float(limerock.inlet_area) * torch.mean(u_f)
        loss_flux = mse(flux_est, torch.tensor(float(volumetric_flow), device=device))

        loss = (
            float(cfg.loss.inlet) * loss_in
            + float(cfg.loss.outlet) * loss_out
            + float(cfg.loss.no_slip) * loss_ns
            + float(cfg.loss.interior) * loss_pde
            + float(cfg.loss.integral_continuity) * loss_flux
        )
        loss.backward()
        optimizer.step()
        scheduler.step()

        if step % int(cfg.training.log_every) == 0:
            print(
                f"step={step:06d} total={loss.item():.4e} inlet={loss_in.item():.4e} "
                f"outlet={loss_out.item():.4e} no_slip={loss_ns.item():.4e} "
                f"interior={loss_pde.item():.4e} flux={loss_flux.item():.4e}"
            )


if __name__ == "__main__":
    run()
