# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hydra
import numpy as np
import torch
from omegaconf import DictConfig as PhysicsNeMoConfig
from torch import nn
from torch.optim import Adam, lr_scheduler

from ldc_v2_utils import (
    FlowNet,
    load_validation,
    ns_residual,
    sample_interior,
    sample_no_slip,
    sample_top_wall,
    sdf_rect,
    weighted_edge_lid_profile,
)


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def run(cfg: PhysicsNeMoConfig) -> None:
    torch.manual_seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    width = float(cfg.domain.width)
    height = float(cfg.domain.height)
    nu = torch.tensor(float(cfg.physics.nu), device=device)

    model = FlowNet(
        in_features=2,
        out_features=3,
        hidden_dim=int(cfg.model.hidden_dim),
        num_layers=int(cfg.model.num_layers),
    ).to(device)

    optimizer = Adam(model.parameters(), lr=float(cfg.optimizer.lr))
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=float(cfg.scheduler.decay_rate))
    mse = nn.MSELoss(reduction="mean")

    val = load_validation(
        "openfoam/cavity_uniformVel0.csv",
        width,
        height,
        {"Points:0": "x", "Points:1": "y", "U:0": "u", "U:1": "v"},
        device,
    )

    for step in range(int(cfg.training.max_steps)):
        optimizer.zero_grad()

        top = sample_top_wall(int(cfg.batch_size.TopWall), width, height, device)
        noslip = sample_no_slip(int(cfg.batch_size.NoSlip), width, height, device)
        interior = sample_interior(int(cfg.batch_size.Interior), width, height, device).requires_grad_(True)

        top_out = model(top)
        u_top = top_out[:, 0:1]
        v_top = top_out[:, 1:2]
        w_top = weighted_edge_lid_profile(top[:, 0:1])
        loss_top = mse(w_top * u_top, w_top * torch.ones_like(u_top)) + mse(v_top, torch.zeros_like(v_top))

        noslip_out = model(noslip)
        loss_noslip = mse(noslip_out[:, 0:1], torch.zeros_like(noslip_out[:, 0:1])) + mse(
            noslip_out[:, 1:2], torch.zeros_like(noslip_out[:, 1:2])
        )

        int_out = model(interior)
        u = int_out[:, 0:1]
        v = int_out[:, 1:2]
        p = int_out[:, 2:3]
        continuity, mx, my, _ = ns_residual(u, v, p, interior, nu.expand_as(u))
        w_int = sdf_rect(interior, width, height)
        loss_int = mse(w_int * continuity, torch.zeros_like(continuity))
        loss_int += mse(w_int * mx, torch.zeros_like(mx))
        loss_int += mse(w_int * my, torch.zeros_like(my))

        loss = (
            float(cfg.loss.top_wall) * loss_top
            + float(cfg.loss.no_slip) * loss_noslip
            + float(cfg.loss.interior) * loss_int
        )
        loss.backward()
        optimizer.step()
        scheduler.step()

        if step % int(cfg.training.log_every) == 0:
            msg = (
                f"step={step:06d} total={loss.item():.4e} top={loss_top.item():.4e} "
                f"noslip={loss_noslip.item():.4e} interior={loss_int.item():.4e}"
            )
            if val:
                with torch.no_grad():
                    xy = torch.cat([val["x"], val["y"]], dim=1)
                    pred = model(xy)
                    val_mse = mse(pred[:, 0:1], val["u"]) + mse(pred[:, 1:2], val["v"])
                    msg += f" val_mse={val_mse.item():.4e}"
            print(msg)


if __name__ == "__main__":
    run()
