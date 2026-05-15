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


def split_domains(points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    lower = points[points[:, 1] < 0.0]
    upper = points[points[:, 1] >= 0.0]
    return lower, upper


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def run(cfg: PhysicsNeMoConfig) -> None:
    torch.manual_seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    width = float(cfg.domain.width)
    height = float(cfg.domain.height)
    nu = torch.tensor(float(cfg.physics.nu), device=device)

    model_1 = FlowNet(2, 3, int(cfg.model.hidden_dim), int(cfg.model.num_layers)).to(device)
    model_2 = FlowNet(2, 3, int(cfg.model.hidden_dim), int(cfg.model.num_layers)).to(device)

    optimizer = Adam(list(model_1.parameters()) + list(model_2.parameters()), lr=float(cfg.optimizer.lr))
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
        interior = sample_interior(int(cfg.batch_size.Interior), width, height, device)
        lower, upper = split_domains(interior)
        lower = lower.requires_grad_(True)
        upper = upper.requires_grad_(True)

        top_2 = model_2(top)
        w_top = weighted_edge_lid_profile(top[:, 0:1])
        loss_top = mse(w_top * top_2[:, 0:1], w_top * torch.ones_like(top_2[:, 0:1])) + mse(
            top_2[:, 1:2], torch.zeros_like(top_2[:, 1:2])
        )

        noslip_1 = model_1(noslip)
        noslip_2 = model_2(noslip)
        loss_noslip = mse(noslip_1[:, 0:2], torch.zeros_like(noslip_1[:, 0:2]))
        loss_noslip += mse(noslip_2[:, 0:2], torch.zeros_like(noslip_2[:, 0:2]))

        out_l = model_1(lower)
        c1, mx1, my1, _ = ns_residual(out_l[:, 0:1], out_l[:, 1:2], out_l[:, 2:3], lower, nu.expand(out_l.shape[0], 1))
        wl = sdf_rect(lower, width, height)
        loss_l = mse(wl * c1, torch.zeros_like(c1)) + mse(wl * mx1, torch.zeros_like(mx1)) + mse(
            wl * my1, torch.zeros_like(my1)
        )

        out_u = model_2(upper)
        c2, mx2, my2, _ = ns_residual(out_u[:, 0:1], out_u[:, 1:2], out_u[:, 2:3], upper, nu.expand(out_u.shape[0], 1))
        wu = sdf_rect(upper, width, height)
        loss_u = mse(wu * c2, torch.zeros_like(c2)) + mse(wu * mx2, torch.zeros_like(mx2)) + mse(
            wu * my2, torch.zeros_like(my2)
        )

        x_if = (torch.rand(int(cfg.batch_size.Interface), 1, device=device) - 0.5) * width
        y_if = torch.zeros_like(x_if)
        interface = torch.cat([x_if, y_if], dim=1)
        p1 = model_1(interface)
        p2 = model_2(interface)
        loss_if = mse(p1[:, 0:1], p2[:, 0:1]) + mse(p1[:, 1:2], p2[:, 1:2]) + mse(
            p1[:, 2:3], p2[:, 2:3]
        )

        loss_int = 0.5 * (loss_l + loss_u)
        loss = (
            float(cfg.loss.top_wall) * loss_top
            + float(cfg.loss.no_slip) * loss_noslip
            + float(cfg.loss.interior) * loss_int
            + float(cfg.loss.interface) * loss_if
        )

        loss.backward()
        optimizer.step()
        scheduler.step()

        if step % int(cfg.training.log_every) == 0:
            msg = (
                f"step={step:06d} total={loss.item():.4e} top={loss_top.item():.4e} "
                f"noslip={loss_noslip.item():.4e} interior={loss_int.item():.4e} interface={loss_if.item():.4e}"
            )
            if val:
                with torch.no_grad():
                    xy = torch.cat([val["x"], val["y"]], dim=1)
                    mask = xy[:, 1:2] < 0.0
                    pred = torch.zeros(xy.shape[0], 3, device=device)
                    pred[mask[:, 0]] = model_1(xy[mask[:, 0]])
                    pred[~mask[:, 0]] = model_2(xy[~mask[:, 0]])
                    val_mse = mse(pred[:, 0:1], val["u"]) + mse(pred[:, 1:2], val["v"])
                    msg += f" val_mse={val_mse.item():.4e}"
            print(msg)


if __name__ == "__main__":
    run()
