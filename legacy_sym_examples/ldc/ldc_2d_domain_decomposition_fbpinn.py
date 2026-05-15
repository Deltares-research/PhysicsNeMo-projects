# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hydra
import matplotlib.pyplot as plt
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


def blend_weights(y: torch.Tensor, beta: float) -> tuple[torch.Tensor, torch.Tensor]:
    b1 = 0.5 * (1.0 - torch.tanh(beta * y))
    b2 = 0.5 * (1.0 + torch.tanh(beta * y))
    return b1, b2


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def run(cfg: PhysicsNeMoConfig) -> None:
    torch.manual_seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    width = float(cfg.domain.width)
    height = float(cfg.domain.height)
    nu = torch.tensor(float(cfg.physics.nu), device=device)
    beta = float(cfg.fbpinn.beta)

    model_1 = FlowNet(2, 3, int(cfg.model.hidden_dim), int(cfg.model.num_layers)).to(device)
    model_2 = FlowNet(2, 3, int(cfg.model.hidden_dim), int(cfg.model.num_layers)).to(device)

    optimizer = Adam(list(model_1.parameters()) + list(model_2.parameters()), lr=float(cfg.optimizer.lr))
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=float(cfg.scheduler.decay_rate))
    mse = nn.MSELoss(reduction="mean")

    y_plot = np.linspace(-height / 2.0, height / 2.0, 100, dtype=np.float32)
    y_t = torch.from_numpy(y_plot.reshape(-1, 1)).to(device)
    with torch.no_grad():
        b1, b2 = blend_weights(y_t, beta)
        plt.figure()
        plt.plot(y_plot, b1.cpu().numpy(), label="basis_function_1", color="blue")
        plt.plot(y_plot, b2.cpu().numpy(), label="basis_function_2", color="green")
        plt.legend()
        plt.savefig("basis_function_viz.png")
        plt.close()

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

        b1_top, b2_top = blend_weights(top[:, 1:2], beta)
        top_1 = model_1(top)
        top_2 = model_2(top)
        top_u = b1_top * top_1[:, 0:1] + b2_top * top_2[:, 0:1]
        top_v = b1_top * top_1[:, 1:2] + b2_top * top_2[:, 1:2]
        w_top = weighted_edge_lid_profile(top[:, 0:1])
        loss_top = mse(w_top * top_u, w_top * torch.ones_like(top_u)) + mse(top_v, torch.zeros_like(top_v))

        b1_ns, b2_ns = blend_weights(noslip[:, 1:2], beta)
        ns_1 = model_1(noslip)
        ns_2 = model_2(noslip)
        ns_u = b1_ns * ns_1[:, 0:1] + b2_ns * ns_2[:, 0:1]
        ns_v = b1_ns * ns_1[:, 1:2] + b2_ns * ns_2[:, 1:2]
        loss_noslip = mse(ns_u, torch.zeros_like(ns_u)) + mse(ns_v, torch.zeros_like(ns_v))

        b1_i, b2_i = blend_weights(interior[:, 1:2], beta)
        out_1 = model_1(interior)
        out_2 = model_2(interior)
        u = b1_i * out_1[:, 0:1] + b2_i * out_2[:, 0:1]
        v = b1_i * out_1[:, 1:2] + b2_i * out_2[:, 1:2]
        p = b1_i * out_1[:, 2:3] + b2_i * out_2[:, 2:3]

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
                    b1v, b2v = blend_weights(xy[:, 1:2], beta)
                    pv1 = model_1(xy)
                    pv2 = model_2(xy)
                    pu = b1v * pv1[:, 0:1] + b2v * pv2[:, 0:1]
                    pv = b1v * pv1[:, 1:2] + b2v * pv2[:, 1:2]
                    val_mse = mse(pu, val["u"]) + mse(pv, val["v"])
                    msg += f" val_mse={val_mse.item():.4e}"
            print(msg)


if __name__ == "__main__":
    run()
