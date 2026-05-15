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
import os
import sys
import warnings

import numpy as np
import torch

from hydra.utils import to_absolute_path
from omegaconf import DictConfig as PhysicsNeMoConfig
from physicsnemo.models.mlp.fully_connected import FullyConnected


def build_inputs(a, x):
    return np.concatenate([a, x], axis=1).astype(np.float32)


def batched_mse(model, x, y, batch_size, device):
    losses = []
    n = x.shape[0]
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        xb = torch.from_numpy(x[start:end]).to(device)
        yb = torch.from_numpy(y[start:end]).to(device)
        pred = model(xb)
        losses.append(torch.mean((pred - yb) ** 2).detach().cpu().item())
    return float(np.mean(losses)) if losses else 0.0


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def run(cfg: PhysicsNeMoConfig) -> None:
    torch.manual_seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = FullyConnected(
        in_features=101,
        out_features=1,
        num_layers=int(cfg.model.num_layers),
        layer_size=int(cfg.model.layer_size),
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.optimizer.lr))
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=float(cfg.scheduler.decay_rate)
    )

    # [datasets]
    # load training data
    file_path = "data/anti_derivative.npy"
    if not os.path.exists(to_absolute_path(file_path)):
        warnings.warn(
            f"Directory {file_path} does not exist. Cannot continue. Please download the additional files from NGC https://catalog.ngc.nvidia.com/orgs/nvidia/teams/physicsnemo/resources/physicsnemo_sym_examples_supplemental_materials"
        )
        sys.exit()

    data = np.load(to_absolute_path(file_path), allow_pickle=True).item()
    x_train = data["x_train"]
    a_train = data["a_train"]
    u_train = data["u_train"]

    # load test data
    x_test = data["x_test"]
    a_test = data["a_test"]
    u_test = data["u_test"]
    # [datasets]

    x_train_full = build_inputs(a_train, x_train)
    y_train_full = u_train.astype(np.float32)
    x_test_full = build_inputs(a_test, x_test)
    y_test_full = u_test.astype(np.float32)
    n_train = x_train_full.shape[0]

    for step in range(int(cfg.training.max_steps)):
        optimizer.zero_grad()

        idx = np.random.randint(0, n_train, size=int(cfg.batch_size.train))
        xb = torch.from_numpy(x_train_full[idx]).to(device)
        yb = torch.from_numpy(y_train_full[idx]).to(device)

        pred = model(xb)
        loss = torch.mean((pred - yb) ** 2)
        loss.backward()
        optimizer.step()
        scheduler.step()

        if step % int(cfg.training.log_every) == 0:
            val_mse = batched_mse(
                model,
                x_test_full,
                y_test_full,
                batch_size=int(cfg.batch_size.validation),
                device=device,
            )
            print(
                f"step={step:06d} train_mse={loss.item():.4e} val_mse={val_mse:.4e}"
            )


if __name__ == "__main__":
    run()
