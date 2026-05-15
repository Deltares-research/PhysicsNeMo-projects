# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

import hydra
import os
import sys
import numpy as np
import torch
from omegaconf import DictConfig as PhysicsNeMoConfig
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh
from torch import nn
from torch.optim import Adam, lr_scheduler

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from compat.waveguide_v2_utils import (
    MLP,
    normal_grad_x_2d,
    sample_line_x_const,
    sample_pec_rect_boundary,
    sample_rect_interior,
)


def laplacian_1d_eig(a, b, n_points, eps_fn, k=1):
    n = n_points - 2
    h = (b - a) / (n_points - 1)
    L = diags([1, -2, 1], [-1, 0, 1], shape=(n, n))
    L = -L / h**2
    y = np.linspace(a, b, num=n_points)
    M = diags([eps_fn(y[1:-1])], [0])
    eigvals, eigvecs = eigsh(L, k=k, M=M, which="SM")
    eigvecs = np.vstack((np.zeros((1, k)), eigvecs, np.zeros((1, k))))
    eigvecs = eigvecs / np.linalg.norm(eigvecs, axis=0, keepdims=True)
    return eigvals.astype(np.float32), eigvecs.astype(np.float32), y.astype(np.float32)


def slab_eps(y: torch.Tensor, height: float, slab_len: float, eps0: float, eps1: float):
    lo = (height - slab_len) / 2.0
    hi = (height + slab_len) / 2.0
    inside = (y > lo) & (y < hi)
    return torch.where(inside, torch.full_like(y, eps1), torch.full_like(y, eps0))


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def run(cfg: PhysicsNeMoConfig) -> None:
    torch.manual_seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    width = float(cfg.domain.width)
    height = float(cfg.domain.height)
    slab_len = float(cfg.physics.slab_length)
    eps0 = float(cfg.physics.eps0)
    eps1 = float(cfg.physics.eps1)
    wave_number = float(cfg.physics.wave_number)

    def eps_numpy(y):
        return np.where(np.logical_and(y > (height - slab_len) / 2, y < (height + slab_len) / 2), eps1, eps0)

    _, eigvecs, ygrid = laplacian_1d_eig(0.0, height, 512, eps_numpy, k=1)

    model = MLP(2, 1, int(cfg.model.hidden_dim), int(cfg.model.num_layers)).to(device)
    mse = nn.MSELoss(reduction="mean")
    optimizer = Adam(model.parameters(), lr=float(cfg.optimizer.lr))
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=float(cfg.scheduler.decay_rate))

    y_port = torch.from_numpy(ygrid.reshape(-1, 1)).to(device)
    port_fixed = torch.cat([torch.zeros_like(y_port), y_port], dim=1)
    target_port = 10.0 * torch.from_numpy(eigvecs[:, 0:1]).to(device)

    for step in range(int(cfg.training.max_steps)):
        optimizer.zero_grad()

        pec = sample_pec_rect_boundary(int(cfg.batch_size.PEC), width, height, device)
        abc = sample_line_x_const(int(cfg.batch_size.ABC), width, height, device)
        interior = sample_rect_interior(int(cfg.batch_size.Interior), width, height, device).requires_grad_(True)

        loss_pec = mse(model(pec), torch.zeros((pec.shape[0], 1), device=device))
        loss_port = mse(model(port_fixed), target_port)

        grad_abc = normal_grad_x_2d(model, abc)
        loss_abc = mse(grad_abc, torch.zeros_like(grad_abc))

        u = model(interior)
        gu = torch.autograd.grad(u, interior, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_x = gu[:, 0:1]
        u_y = gu[:, 1:2]
        u_xx = torch.autograd.grad(u_x, interior, grad_outputs=torch.ones_like(u_x), create_graph=True)[0][:, 0:1]
        u_yy = torch.autograd.grad(u_y, interior, grad_outputs=torch.ones_like(u_y), create_graph=True)[0][:, 1:2]
        eps_y = slab_eps(interior[:, 1:2], height, slab_len, eps0, eps1)
        resid = u_xx + u_yy + (wave_number**2) * eps_y * u
        loss_int = mse(resid, torch.zeros_like(resid))

        loss = (
            float(cfg.loss.pec) * loss_pec
            + float(cfg.loss.port) * loss_port
            + float(cfg.loss.abc) * loss_abc
            + float(cfg.loss.interior) * loss_int
        )
        loss.backward()
        optimizer.step()
        scheduler.step()

        if step % int(cfg.training.log_every) == 0:
            print(
                f"step={step:06d} total={loss.item():.4e} pec={loss_pec.item():.4e} "
                f"port={loss_port.item():.4e} abc={loss_abc.item():.4e} interior={loss_int.item():.4e}"
            )


if __name__ == "__main__":
    run()
