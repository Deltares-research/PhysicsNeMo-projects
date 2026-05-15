# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, Tuple

import numpy as np
import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, in_features: int, out_features: int, hidden_dim: int, num_layers: int):
        super().__init__()
        layers = [nn.Linear(in_features, hidden_dim), nn.Tanh()]
        for _ in range(max(0, num_layers - 1)):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
        layers.append(nn.Linear(hidden_dim, out_features))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def sample_rect_interior(n: int, width: float, height: float, device: torch.device) -> torch.Tensor:
    x = torch.rand(n, 1, device=device) * width
    y = torch.rand(n, 1, device=device) * height
    return torch.cat([x, y], dim=1)


def sample_box_interior(
    n: int, width: float, length: float, height: float, device: torch.device
) -> torch.Tensor:
    x = torch.rand(n, 1, device=device) * width
    y = torch.rand(n, 1, device=device) * length
    z = torch.rand(n, 1, device=device) * height
    return torch.cat([x, y, z], dim=1)


def sample_line_x_const(n: int, x_val: float, y_max: float, device: torch.device) -> torch.Tensor:
    x = torch.full((n, 1), float(x_val), device=device)
    y = torch.rand(n, 1, device=device) * y_max
    return torch.cat([x, y], dim=1)


def sample_plane_x_const(
    n: int, x_val: float, y_max: float, z_max: float, device: torch.device
) -> torch.Tensor:
    x = torch.full((n, 1), float(x_val), device=device)
    y = torch.rand(n, 1, device=device) * y_max
    z = torch.rand(n, 1, device=device) * z_max
    return torch.cat([x, y, z], dim=1)


def sample_pec_rect_boundary(n: int, width: float, height: float, device: torch.device) -> torch.Tensor:
    n2 = max(1, n // 2)
    x = torch.rand(n2, 1, device=device) * width
    y0 = torch.zeros((n2, 1), device=device)
    yh = torch.full((n2, 1), height, device=device)
    return torch.cat([torch.cat([x, y0], dim=1), torch.cat([x, yh], dim=1)], dim=0)


def sample_pec_box_sidewalls(
    n: int, width: float, length: float, height: float, device: torch.device
) -> torch.Tensor:
    n3 = max(1, n // 3)
    x = torch.rand(n3, 1, device=device) * width
    y = torch.rand(n3, 1, device=device) * length
    z = torch.rand(n3, 1, device=device) * height

    y0 = torch.zeros((n3, 1), device=device)
    yl = torch.full((n3, 1), length, device=device)
    z0 = torch.zeros((n3, 1), device=device)
    zh = torch.full((n3, 1), height, device=device)

    p1 = torch.cat([x, y0, z], dim=1)
    p2 = torch.cat([x, yl, z], dim=1)
    p3 = torch.cat([x, y, z0], dim=1)
    p4 = torch.cat([x, y, zh], dim=1)
    return torch.cat([p1, p2, p3, p4], dim=0)


def helmholtz_residual_2d(model: nn.Module, xy: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    xy = xy.requires_grad_(True)
    u = model(xy)
    gu = torch.autograd.grad(u, xy, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_x = gu[:, 0:1]
    u_y = gu[:, 1:2]
    u_xx = torch.autograd.grad(u_x, xy, grad_outputs=torch.ones_like(u_x), create_graph=True)[0][:, 0:1]
    u_yy = torch.autograd.grad(u_y, xy, grad_outputs=torch.ones_like(u_y), create_graph=True)[0][:, 1:2]
    return u_xx + u_yy + (k**2) * u


def normal_grad_x_2d(model: nn.Module, xy: torch.Tensor) -> torch.Tensor:
    xy = xy.requires_grad_(True)
    u = model(xy)
    gu = torch.autograd.grad(u, xy, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    return gu[:, 0:1]


def vector_helmholtz_residual_3d(
    model: nn.Module, xyz: torch.Tensor, k: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    xyz = xyz.requires_grad_(True)
    out = model(xyz)
    ux = out[:, 0:1]
    uy = out[:, 1:2]
    uz = out[:, 2:3]

    def lap(v: torch.Tensor) -> torch.Tensor:
        gv = torch.autograd.grad(v, xyz, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        v_x = gv[:, 0:1]
        v_y = gv[:, 1:2]
        v_z = gv[:, 2:3]
        v_xx = torch.autograd.grad(v_x, xyz, grad_outputs=torch.ones_like(v_x), create_graph=True)[0][:, 0:1]
        v_yy = torch.autograd.grad(v_y, xyz, grad_outputs=torch.ones_like(v_y), create_graph=True)[0][:, 1:2]
        v_zz = torch.autograd.grad(v_z, xyz, grad_outputs=torch.ones_like(v_z), create_graph=True)[0][:, 2:3]
        return v_xx + v_yy + v_zz

    return lap(ux) + (k**2) * ux, lap(uy) + (k**2) * uy, lap(uz) + (k**2) * uz


def normal_grad_x_3d(model: nn.Module, xyz: torch.Tensor) -> torch.Tensor:
    xyz = xyz.requires_grad_(True)
    out = model(xyz)
    grads = []
    for i in range(3):
        comp = out[:, i : i + 1]
        g = torch.autograd.grad(comp, xyz, grad_outputs=torch.ones_like(comp), create_graph=True)[0]
        grads.append(g[:, 0:1])
    return torch.cat(grads, dim=1)


def read_csv_columns(file_path: str, mapping: Dict[str, str]) -> Dict[str, np.ndarray]:
    data = np.genfromtxt(file_path, delimiter=",", names=True)
    if data is None:
        return {}
    out: Dict[str, np.ndarray] = {}
    for src, dst in mapping.items():
        out[dst] = np.asarray(data[src], dtype=np.float32).reshape(-1, 1)
    return out
