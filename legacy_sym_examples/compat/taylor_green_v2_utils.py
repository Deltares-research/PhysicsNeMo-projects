# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import torch
from torch import nn


class TGNet(nn.Module):
    def __init__(self, in_features: int, out_features: int, hidden_dim: int, num_layers: int):
        super().__init__()
        layers = [nn.Linear(in_features, hidden_dim), nn.Tanh()]
        for _ in range(max(0, num_layers - 1)):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
        layers.append(nn.Linear(hidden_dim, out_features))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def sample_space_time(n: int, t_min: float, t_max: float, device: torch.device) -> torch.Tensor:
    xyz = torch.rand(n, 3, device=device) * (2.0 * np.pi)
    t = torch.rand(n, 1, device=device) * (t_max - t_min) + t_min
    return torch.cat([xyz, t], dim=1)


def sample_space(n: int, device: torch.device) -> torch.Tensor:
    return torch.rand(n, 3, device=device) * (2.0 * np.pi)


def exact_initial(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor):
    u = torch.sin(x) * torch.cos(y) * torch.cos(z)
    v = -torch.cos(x) * torch.sin(y) * torch.cos(z)
    w = torch.zeros_like(u)
    p = (1.0 / 16.0) * (torch.cos(2.0 * x) + torch.cos(2.0 * y)) * (torch.cos(2.0 * z) + 2.0)
    return u, v, w, p


def ns_residual_time3d(
    out: torch.Tensor, xytz: torch.Tensor, nu: float
):
    u = out[:, 0:1]
    v = out[:, 1:2]
    w = out[:, 2:3]
    p = out[:, 3:4]

    gu = torch.autograd.grad(u, xytz, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    gv = torch.autograd.grad(v, xytz, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    gw = torch.autograd.grad(w, xytz, grad_outputs=torch.ones_like(w), create_graph=True)[0]
    gp = torch.autograd.grad(p, xytz, grad_outputs=torch.ones_like(p), create_graph=True)[0]

    u_x, u_y, u_z, u_t = gu[:, 0:1], gu[:, 1:2], gu[:, 2:3], gu[:, 3:4]
    v_x, v_y, v_z, v_t = gv[:, 0:1], gv[:, 1:2], gv[:, 2:3], gv[:, 3:4]
    w_x, w_y, w_z, w_t = gw[:, 0:1], gw[:, 1:2], gw[:, 2:3], gw[:, 3:4]
    p_x, p_y, p_z = gp[:, 0:1], gp[:, 1:2], gp[:, 2:3]

    def lap(gx, gy, gz):
        gxx = torch.autograd.grad(gx, xytz, grad_outputs=torch.ones_like(gx), create_graph=True)[0][:, 0:1]
        gyy = torch.autograd.grad(gy, xytz, grad_outputs=torch.ones_like(gy), create_graph=True)[0][:, 1:2]
        gzz = torch.autograd.grad(gz, xytz, grad_outputs=torch.ones_like(gz), create_graph=True)[0][:, 2:3]
        return gxx + gyy + gzz

    continuity = u_x + v_y + w_z
    mom_x = u_t + u * u_x + v * u_y + w * u_z + p_x - nu * lap(u_x, u_y, u_z)
    mom_y = v_t + u * v_x + v * v_y + w * v_z + p_y - nu * lap(v_x, v_y, v_z)
    mom_z = w_t + u * w_x + v * w_y + w * w_z + p_z - nu * lap(w_x, w_y, w_z)
    return continuity, mom_x, mom_y, mom_z
