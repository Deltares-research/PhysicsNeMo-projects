# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import hydra
import numpy as np
import torch
from omegaconf import DictConfig as PhysicsNeMoConfig
from torch import nn
from torch.optim import Adam, lr_scheduler

from compat.waveguide_v2_utils import (
    MLP,
    helmholtz_residual_2d,
    normal_grad_x_2d,
    read_csv_columns,
    sample_line_x_const,
    sample_pec_rect_boundary,
    sample_rect_interior,
)


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def run(cfg: PhysicsNeMoConfig) -> None:
    torch.manual_seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    width = float(cfg.domain.width)
    height = float(cfg.domain.height)
    wave_number = float(cfg.physics.wave_number)
    mode = int(cfg.physics.mode)

    model = MLP(2, 1, int(cfg.model.hidden_dim), int(cfg.model.num_layers)).to(device)
    mse = nn.MSELoss(reduction="mean")
    optimizer = Adam(model.parameters(), lr=float(cfg.optimizer.lr))
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=float(cfg.scheduler.decay_rate))

    val = {}
    val_path = "../validation/2Dwaveguide_32_2.csv"
    if os.path.exists(val_path):
        val = read_csv_columns(val_path, {"x": "x", "y": "y", "u": "u"})

    for step in range(int(cfg.training.max_steps)):
        optimizer.zero_grad()

        pec = sample_pec_rect_boundary(int(cfg.batch_size.PEC), width, height, device)
        port = sample_line_x_const(int(cfg.batch_size.Waveguide_port), 0.0, height, device)
        abc = sample_line_x_const(int(cfg.batch_size.ABC), width, height, device)
        interior = sample_rect_interior(int(cfg.batch_size.Interior), width, height, device)

        u_pec = model(pec)
        loss_pec = mse(u_pec, torch.zeros_like(u_pec))

        y = port[:, 1:2]
        target_port = torch.sin(mode * torch.pi * y / height)
        loss_port = mse(model(port), target_port)

        grad_out = normal_grad_x_2d(model, abc)
        loss_abc = mse(grad_out, torch.zeros_like(grad_out))

        k = torch.full((interior.shape[0], 1), wave_number, device=device)
        res = helmholtz_residual_2d(model, interior, k)
        loss_int = mse(res, torch.zeros_like(res))

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
            msg = (
                f"step={step:06d} total={loss.item():.4e} pec={loss_pec.item():.4e} "
                f"port={loss_port.item():.4e} abc={loss_abc.item():.4e} interior={loss_int.item():.4e}"
            )
            if val:
                with torch.no_grad():
                    xy = torch.from_numpy(np.concatenate([val["x"], val["y"]], axis=1)).to(device)
                    true_u = torch.from_numpy(val["u"]).to(device)
                    val_mse = mse(model(xy), true_u)
                    msg += f" val_mse={val_mse.item():.4e}"
            print(msg)


if __name__ == "__main__":
    run()
