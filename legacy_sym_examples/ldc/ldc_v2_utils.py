# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import csv
import os
from typing import Dict, Tuple

import numpy as np
import torch
from torch import nn


class FlowNet(nn.Module):
    def __init__(self, in_features: int, out_features: int, hidden_dim: int, num_layers: int):
        super().__init__()
        layers = [nn.Linear(in_features, hidden_dim), nn.Tanh()]
        for _ in range(max(0, num_layers - 1)):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
        layers.append(nn.Linear(hidden_dim, out_features))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def sample_interior(n: int, width: float, height: float, device: torch.device) -> torch.Tensor:
    xy = torch.rand(n, 2, device=device)
    xy[:, 0] = (xy[:, 0] - 0.5) * width
    xy[:, 1] = (xy[:, 1] - 0.5) * height
    return xy


def sample_top_wall(n: int, width: float, height: float, device: torch.device) -> torch.Tensor:
    x = (torch.rand(n, 1, device=device) - 0.5) * width
    y = torch.full((n, 1), height / 2.0, device=device)
    return torch.cat([x, y], dim=1)


def sample_no_slip(n: int, width: float, height: float, device: torch.device) -> torch.Tensor:
    n_side = max(1, n // 3)
    x_rand = (torch.rand(n_side, 1, device=device) - 0.5) * width
    y_bottom = torch.full((n_side, 1), -height / 2.0, device=device)

    y_rand = (torch.rand(n_side, 1, device=device) - 0.5) * height
    x_left = torch.full((n_side, 1), -width / 2.0, device=device)
    x_right = torch.full((n_side, 1), width / 2.0, device=device)

    bottom = torch.cat([x_rand, y_bottom], dim=1)
    left = torch.cat([x_left, y_rand], dim=1)
    right = torch.cat([x_right, y_rand], dim=1)
    return torch.cat([bottom, left, right], dim=0)


def sdf_rect(coords: torch.Tensor, width: float, height: float) -> torch.Tensor:
    dx = width / 2.0 - torch.abs(coords[:, 0:1])
    dy = height / 2.0 - torch.abs(coords[:, 1:2])
    return torch.clamp(torch.minimum(dx, dy), min=0.0)


def ns_residual(
    u: torch.Tensor,
    v: torch.Tensor,
    p: torch.Tensor,
    coords: torch.Tensor,
    nu: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    ones_u = torch.ones_like(u)
    ones_v = torch.ones_like(v)
    ones_p = torch.ones_like(p)

    gu = torch.autograd.grad(u, coords, grad_outputs=ones_u, create_graph=True)[0]
    gv = torch.autograd.grad(v, coords, grad_outputs=ones_v, create_graph=True)[0]
    gp = torch.autograd.grad(p, coords, grad_outputs=ones_p, create_graph=True)[0]

    u_x = gu[:, 0:1]
    u_y = gu[:, 1:2]
    v_x = gv[:, 0:1]
    v_y = gv[:, 1:2]
    p_x = gp[:, 0:1]
    p_y = gp[:, 1:2]

    u_xx = torch.autograd.grad(u_x, coords, grad_outputs=torch.ones_like(u_x), create_graph=True)[0][:, 0:1]
    u_yy = torch.autograd.grad(u_y, coords, grad_outputs=torch.ones_like(u_y), create_graph=True)[0][:, 1:2]
    v_xx = torch.autograd.grad(v_x, coords, grad_outputs=torch.ones_like(v_x), create_graph=True)[0][:, 0:1]
    v_yy = torch.autograd.grad(v_y, coords, grad_outputs=torch.ones_like(v_y), create_graph=True)[0][:, 1:2]

    continuity = u_x + v_y
    momentum_x = u * u_x + v * u_y + p_x - nu * (u_xx + u_yy)
    momentum_y = u * v_x + v * v_y + p_y - nu * (v_xx + v_yy)

    grads = {
        "u_x": u_x,
        "u_y": u_y,
        "v_x": v_x,
        "v_y": v_y,
    }
    return continuity, momentum_x, momentum_y, grads


def weighted_edge_lid_profile(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(1.0 - 20.0 * torch.abs(x), min=0.0)


def read_openfoam_csv(file_path: str, mapping: Dict[str, str]) -> Dict[str, np.ndarray]:
    if not os.path.exists(file_path):
        return {}

    out: Dict[str, list[float]] = {}
    for v in mapping.values():
        out[v] = []

    with open(file_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for src, dst in mapping.items():
                out[dst].append(float(row[src]))

    return {k: np.asarray(v, dtype=np.float32).reshape(-1, 1) for k, v in out.items()}


def load_validation(
    file_path: str,
    width: float,
    height: float,
    mapping: Dict[str, str],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    data = read_openfoam_csv(file_path, mapping)
    if not data:
        return {}

    if "x" in data:
        data["x"] = data["x"] - (width / 2.0)
    if "y" in data:
        data["y"] = data["y"] - (height / 2.0)

    out: Dict[str, torch.Tensor] = {}
    for k, v in data.items():
        out[k] = torch.from_numpy(v).to(device)
    return out
