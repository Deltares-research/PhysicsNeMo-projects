# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

import hydra
import os
import sys
import numpy as np
import torch
from omegaconf import DictConfig as PhysicsNeMoConfig
from torch import nn
from torch.optim import Adam, lr_scheduler

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from compat.taylor_green_v2_utils import TGNet, exact_initial, ns_residual_time3d, sample_space, sample_space_time


def causal_interior_loss(
    model: nn.Module,
    n_points: int,
    n_chunks: int,
    t0: float,
    t1: float,
    nu: float,
    mse: nn.Module,
    device: torch.device,
):
    dt = (t1 - t0) / float(n_chunks)
    chunk_losses = []
    for i in range(n_chunks):
        ta = t0 + i * dt
        tb = ta + dt
        pts = sample_space_time(n_points, ta, tb, device).requires_grad_(True)
        res = ns_residual_time3d(model(pts), pts, nu)
        chunk_losses.append(sum(mse(r, torch.zeros_like(r)) for r in res))

    weights = []
    accum = torch.tensor(0.0, device=device)
    alpha = 10.0
    for l in chunk_losses:
        w = torch.exp(-alpha * accum.detach())
        weights.append(w)
        accum = accum + l

    loss = torch.tensor(0.0, device=device)
    for w, l in zip(weights, chunk_losses):
        loss = loss + w * l
    return loss


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def run(cfg: PhysicsNeMoConfig) -> None:
    torch.manual_seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    nu = float(cfg.physics.nu)
    time_window = float(cfg.physics.time_window)
    nr_time_windows = int(cfg.physics.nr_time_windows)

    model = TGNet(4, 4, int(cfg.model.hidden_dim), int(cfg.model.num_layers)).to(device)
    mse = nn.MSELoss(reduction="mean")
    optimizer = Adam(model.parameters(), lr=float(cfg.optimizer.lr))
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=float(cfg.scheduler.decay_rate))

    for win in range(nr_time_windows):
        t0 = float(win) * time_window
        t1 = float(win + 1) * time_window
        for step in range(int(cfg.training.max_steps_per_window)):
            optimizer.zero_grad()

            xyz0 = sample_space(int(cfg.batch_size.initial_condition), device)
            t0_col = torch.full((xyz0.shape[0], 1), t0, device=device)
            ic_in = torch.cat([xyz0, t0_col], dim=1)
            ic_out = model(ic_in)
            u0, v0, w0, p0 = exact_initial(xyz0[:, 0:1], xyz0[:, 1:2], xyz0[:, 2:3])
            decay = torch.exp(torch.tensor(-2.0 * nu * t0, device=device))
            loss_ic = mse(ic_out[:, 0:1], u0 * decay)
            loss_ic += mse(ic_out[:, 1:2], v0 * decay)
            loss_ic += mse(ic_out[:, 2:3], w0)
            loss_ic += mse(ic_out[:, 3:4], p0 * (decay**2))

            loss_int = causal_interior_loss(
                model,
                int(cfg.batch_size.interior),
                int(cfg.causal.n_chunks),
                t0,
                t1,
                nu,
                mse,
                device,
            )

            loss = float(cfg.loss.initial_condition) * loss_ic + float(cfg.loss.interior) * loss_int
            loss.backward()
            optimizer.step()
            scheduler.step()

            if step % int(cfg.training.log_every) == 0:
                print(
                    f"window={win:02d} step={step:06d} total={loss.item():.4e} "
                    f"ic={loss_ic.item():.4e} interior_causal={loss_int.item():.4e}"
                )


if __name__ == "__main__":
    run()
