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


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def run(cfg: PhysicsNeMoConfig) -> None:
    torch.manual_seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = FullyConnected(
        in_features=1,
        out_features=3,
        num_layers=int(cfg.model.num_layers),
        layer_size=int(cfg.model.layer_size),
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.optimizer.lr))
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=float(cfg.scheduler.decay_rate)
    )

    k1, k2, k3, k4 = 2.0, 1.0, 1.0, 2.0
    m1, m2, m3 = 1.0, 1.0, 1.0
    t_max = 10.0

    for step in range(int(cfg.training.max_steps)):
        optimizer.zero_grad()

        t_int = torch.rand(int(cfg.batch_size.interior), 1, device=device) * t_max
        t_int.requires_grad_(True)
        x_int = model(t_int)
        x1 = x_int[:, 0:1]
        x2 = x_int[:, 1:2]
        x3 = x_int[:, 2:3]

        dx1 = torch.autograd.grad(
            x1, t_int, grad_outputs=torch.ones_like(x1), create_graph=True
        )[0]
        dx2 = torch.autograd.grad(
            x2, t_int, grad_outputs=torch.ones_like(x2), create_graph=True
        )[0]
        dx3 = torch.autograd.grad(
            x3, t_int, grad_outputs=torch.ones_like(x3), create_graph=True
        )[0]
        d2x1 = torch.autograd.grad(
            dx1, t_int, grad_outputs=torch.ones_like(dx1), create_graph=True
        )[0]
        d2x2 = torch.autograd.grad(
            dx2, t_int, grad_outputs=torch.ones_like(dx2), create_graph=True
        )[0]
        d2x3 = torch.autograd.grad(
            dx3, t_int, grad_outputs=torch.ones_like(dx3), create_graph=True
        )[0]

        ode1 = m1 * d2x1 + k1 * x1 - k2 * (x2 - x1)
        ode2 = m2 * d2x2 + k2 * (x2 - x1) - k3 * (x3 - x2)
        ode3 = m3 * d2x3 + k3 * (x3 - x2) + k4 * x3
        ode_loss = torch.mean(ode1**2) + torch.mean(ode2**2) + torch.mean(ode3**2)

        t0 = torch.zeros(int(cfg.batch_size.IC), 1, device=device, requires_grad=True)
        x0 = model(t0)
        x10 = x0[:, 0:1]
        x20 = x0[:, 1:2]
        x30 = x0[:, 2:3]
        dx10 = torch.autograd.grad(
            x10, t0, grad_outputs=torch.ones_like(x10), create_graph=True
        )[0]
        dx20 = torch.autograd.grad(
            x20, t0, grad_outputs=torch.ones_like(x20), create_graph=True
        )[0]
        dx30 = torch.autograd.grad(
            x30, t0, grad_outputs=torch.ones_like(x30), create_graph=True
        )[0]
        ic_loss = (
            torch.mean((x10 - 1.0) ** 2)
            + torch.mean(x20**2)
            + torch.mean(x30**2)
            + torch.mean(dx10**2)
            + torch.mean(dx20**2)
            + torch.mean(dx30**2)
        )

        loss = float(cfg.loss_weights.ic) * ic_loss + float(cfg.loss_weights.ode) * ode_loss
        loss.backward()
        optimizer.step()
        scheduler.step()

        if step % int(cfg.training.log_every) == 0:
            with torch.no_grad():
                t = np.arange(0.0, t_max, 0.001, dtype=np.float32)[:, None]
                t_t = torch.from_numpy(t).to(device)
                pred = model(t_t).cpu().numpy()
                x1_true = (1.0 / 6.0) * np.cos(t) + 0.5 * np.cos(np.sqrt(3.0) * t) + (1.0 / 3.0) * np.cos(2.0 * t)
                rel_l2 = np.linalg.norm(pred[:, 0:1] - x1_true) / (
                    np.linalg.norm(x1_true) + 1e-12
                )
                print(
                    f"step={step:06d} loss={loss.item():.4e} ic={ic_loss.item():.4e} "
                    f"ode={ode_loss.item():.4e} rel_l2_x1={rel_l2:.4e}"
                )


if __name__ == "__main__":
    run()
