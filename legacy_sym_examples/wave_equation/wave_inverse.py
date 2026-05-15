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
import torch
import numpy as np
import torch.nn.functional as F

from omegaconf import DictConfig as PhysicsNeMoConfig
from physicsnemo.models.mlp.fully_connected import FullyConnected


def wave_residual_inverse(coords, u_pred, c_pred):
    du = torch.autograd.grad(
        u_pred,
        coords,
        grad_outputs=torch.ones_like(u_pred),
        create_graph=True,
    )[0]
    u_x = du[:, 0:1]
    u_t = du[:, 1:2]

    d2u_dt = torch.autograd.grad(
        u_t,
        coords,
        grad_outputs=torch.ones_like(u_t),
        create_graph=True,
    )[0]
    u_tt = d2u_dt[:, 1:2]

    flux = (c_pred**2) * u_x
    dflux = torch.autograd.grad(
        flux,
        coords,
        grad_outputs=torch.ones_like(flux),
        create_graph=True,
    )[0]
    flux_x = dflux[:, 0:1]
    return u_tt - flux_x


@hydra.main(version_base="1.3", config_path="conf", config_name="config_inverse")
def run(cfg: PhysicsNeMoConfig) -> None:
    torch.manual_seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wave_net = FullyConnected(
        in_features=2,
        out_features=1,
        num_layers=int(cfg.model.num_layers),
        layer_size=int(cfg.model.layer_size),
    ).to(device)
    invert_net = FullyConnected(
        in_features=2,
        out_features=1,
        num_layers=int(cfg.model.num_layers),
        layer_size=int(cfg.model.layer_size),
    ).to(device)

    optimizer = torch.optim.Adam(
        list(wave_net.parameters()) + list(invert_net.parameters()),
        lr=float(cfg.optimizer.lr),
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=float(cfg.scheduler.decay_rate)
    )

    L = float(np.pi)
    deltaT = 0.01
    deltaX = 0.01
    x = np.arange(0, L, deltaX)
    t = np.arange(0, 2 * L, deltaT)
    X, T = np.meshgrid(x, t)
    X = np.expand_dims(X.flatten(), axis=-1)
    T = np.expand_dims(T.flatten(), axis=-1)
    u = np.sin(X) * (np.cos(T) + np.sin(T))
    x_full = torch.from_numpy(X.astype(np.float32)).to(device)
    t_full = torch.from_numpy(T.astype(np.float32)).to(device)
    u_full = torch.from_numpy(u.astype(np.float32)).to(device)
    n_total = x_full.shape[0]

    for step in range(int(cfg.training.max_steps)):
        optimizer.zero_grad()

        idx = torch.randint(0, n_total, (int(cfg.batch_size.data),), device=device)
        x_b = x_full[idx]
        t_b = t_full[idx]
        u_b = u_full[idx]

        coords = torch.cat([x_b, t_b], dim=1).requires_grad_(True)
        u_pred = wave_net(coords)
        c_raw = invert_net(coords)
        c_pred = F.softplus(c_raw) + float(cfg.model.min_speed)

        data_loss = torch.mean((u_pred - u_b) ** 2)
        wave_residual = wave_residual_inverse(coords, u_pred, c_pred)
        pde_loss = torch.mean(wave_residual**2)

        loss = data_loss + float(cfg.loss_weights.pde) * pde_loss
        loss.backward()
        optimizer.step()
        scheduler.step()

        if step % int(cfg.training.log_every) == 0:
            with torch.no_grad():
                mean_c = c_pred.mean().item()
                print(
                    f"step={step:06d} loss={loss.item():.4e} "
                    f"data={data_loss.item():.4e} pde={pde_loss.item():.4e} mean_c={mean_c:.5f}"
                )


if __name__ == "__main__":
    run()
