# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import hydra
import numpy as np
import torch

from omegaconf import DictConfig as PhysicsNeMoConfig
from physicsnemo.models.mlp.fully_connected import FullyConnected


def wave_residual_1d(model, coords, c=1.0):
    u = model(coords)
    du = torch.autograd.grad(
        u,
        coords,
        grad_outputs=torch.ones_like(u),
        create_graph=True,
    )[0]
    u_x = du[:, 0:1]
    u_t = du[:, 1:2]

    d2u_dx = torch.autograd.grad(
        u_x,
        coords,
        grad_outputs=torch.ones_like(u_x),
        create_graph=True,
    )[0]
    d2u_dt = torch.autograd.grad(
        u_t,
        coords,
        grad_outputs=torch.ones_like(u_t),
        create_graph=True,
    )[0]

    u_xx = d2u_dx[:, 0:1]
    u_tt = d2u_dt[:, 1:2]
    return u_tt - (c**2) * u_xx


@hydra.main(version_base="1.3", config_path="conf", config_name="config_causal")
def run(cfg: PhysicsNeMoConfig) -> None:
    torch.manual_seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = FullyConnected(
        in_features=2,
        out_features=1,
        num_layers=int(cfg.model.num_layers),
        layer_size=int(cfg.model.layer_size),
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.optimizer.lr))
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=float(cfg.scheduler.decay_rate)
    )

    L = float(np.pi)
    t_max = 4.0 * L

    for step in range(int(cfg.training.max_steps)):
        optimizer.zero_grad()

        if step < int(cfg.training.curriculum_steps):
            active_t_max = t_max * (step + 1) / float(cfg.training.curriculum_steps)
        else:
            active_t_max = t_max

        x_int = torch.rand(int(cfg.batch_size.interior), 1, device=device) * L
        t_int = torch.rand(int(cfg.batch_size.interior), 1, device=device) * active_t_max
        coords_int = torch.cat([x_int, t_int], dim=1).requires_grad_(True)
        wave_residual = wave_residual_1d(model, coords_int, c=1.0)
        pde_loss = torch.mean(wave_residual**2)

        x_ic = torch.rand(int(cfg.batch_size.IC), 1, device=device) * L
        t_ic = torch.zeros_like(x_ic, requires_grad=True)
        coords_ic = torch.cat([x_ic, t_ic], dim=1)
        u_ic = model(coords_ic)
        du_dt_ic = torch.autograd.grad(
            u_ic,
            t_ic,
            grad_outputs=torch.ones_like(u_ic),
            create_graph=True,
        )[0]
        u_ic_target = torch.sin(x_ic)
        ic_loss = torch.mean((u_ic - u_ic_target) ** 2) + torch.mean(
            (du_dt_ic - u_ic_target) ** 2
        )

        t_bc = torch.rand(int(cfg.batch_size.BC), 1, device=device) * active_t_max
        x_left = torch.zeros_like(t_bc)
        x_right = torch.full_like(t_bc, L)
        u_bc_left = model(torch.cat([x_left, t_bc], dim=1))
        u_bc_right = model(torch.cat([x_right, t_bc], dim=1))
        bc_loss = torch.mean(u_bc_left**2) + torch.mean(u_bc_right**2)

        loss = (
            float(cfg.loss_weights.ic) * ic_loss
            + float(cfg.loss_weights.bc) * bc_loss
            + float(cfg.loss_weights.pde) * pde_loss
        )
        loss.backward()
        optimizer.step()
        scheduler.step()

        if step % int(cfg.training.log_every) == 0:
            with torch.no_grad():
                x_eval = np.arange(0, L, 0.01, dtype=np.float32)
                t_eval = np.arange(0, t_max, 0.01, dtype=np.float32)
                xx, tt = np.meshgrid(x_eval, t_eval)
                x_eval_t = torch.from_numpy(xx.reshape(-1, 1)).to(device)
                t_eval_t = torch.from_numpy(tt.reshape(-1, 1)).to(device)
                u_pred = model(torch.cat([x_eval_t, t_eval_t], dim=1)).cpu().numpy()
                u_true = np.sin(x_eval_t.cpu().numpy()) * (
                    np.cos(t_eval_t.cpu().numpy()) + np.sin(t_eval_t.cpu().numpy())
                )
                rel_l2 = np.linalg.norm(u_pred - u_true) / (
                    np.linalg.norm(u_true) + 1e-12
                )
                print(
                    f"step={step:06d} active_t_max={active_t_max:.4f} "
                    f"loss={loss.item():.4e} pde={pde_loss.item():.4e} "
                    f"ic={ic_loss.item():.4e} bc={bc_loss.item():.4e} rel_l2={rel_l2:.4e}"
                )


if __name__ == "__main__":
    run()
