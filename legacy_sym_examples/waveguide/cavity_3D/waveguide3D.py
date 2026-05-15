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

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from compat.waveguide_v2_utils import (
    MLP,
    normal_grad_x_3d,
    sample_box_interior,
    sample_pec_box_sidewalls,
    sample_plane_x_const,
    vector_helmholtz_residual_3d,
)


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def run(cfg: PhysicsNeMoConfig) -> None:
    torch.manual_seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    width = float(cfg.domain.width)
    length = float(cfg.domain.length)
    height = float(cfg.domain.height)
    wave_number = float(cfg.physics.wave_number)
    mode = int(cfg.physics.mode)

    model = MLP(3, 3, int(cfg.model.hidden_dim), int(cfg.model.num_layers)).to(device)
    mse = nn.MSELoss(reduction="mean")
    optimizer = Adam(model.parameters(), lr=float(cfg.optimizer.lr))
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=float(cfg.scheduler.decay_rate))

    for step in range(int(cfg.training.max_steps)):
        optimizer.zero_grad()

        pec = sample_pec_box_sidewalls(int(cfg.batch_size.PEC), width, length, height, device)
        port = sample_plane_x_const(int(cfg.batch_size.Waveguide_port), 0.0, length, height, device)
        abc = sample_plane_x_const(int(cfg.batch_size.ABC), width, length, height, device)
        interior = sample_box_interior(int(cfg.batch_size.Interior), width, length, height, device)

        out_pec = model(pec)
        loss_pec = mse(out_pec, torch.zeros_like(out_pec))

        y = port[:, 1:2]
        z = port[:, 2:3]
        target_uz = torch.sin(mode * torch.pi * y / length) * torch.sin(mode * torch.pi * z / height)
        out_port = model(port)
        loss_port = mse(out_port[:, 2:3], target_uz)

        grad_x = normal_grad_x_3d(model, abc)
        loss_abc = mse(grad_x, torch.zeros_like(grad_x))

        k = torch.full((interior.shape[0], 1), wave_number, device=device)
        rx, ry, rz = vector_helmholtz_residual_3d(model, interior, k)
        loss_int = mse(rx, torch.zeros_like(rx)) + mse(ry, torch.zeros_like(ry)) + mse(
            rz, torch.zeros_like(rz)
        )

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
