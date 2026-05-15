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
import os
import torch
from omegaconf import DictConfig as PhysicsNeMoConfig
from torch.nn import MSELoss
from torch.optim import Adam, lr_scheduler
from physicsnemo.models.mlp.fully_connected import FullyConnected

from utilities import download_FNO_dataset, load_deeponet_dataset


def ensure_dataset(cfg: PhysicsNeMoConfig) -> tuple[str, str]:
    dataset_name = str(cfg.data.dataset_name)
    dataset_dir = str(cfg.data.dataset_dir)
    file_train = os.path.join(dataset_dir, dataset_name, "piececonst_r241_N1024_smooth1.hdf5")
    file_test = os.path.join(dataset_dir, dataset_name, "piececonst_r241_N1024_smooth2.hdf5")

    if bool(cfg.data.auto_download):
        download_FNO_dataset(dataset_name, outdir=dataset_dir)

    if not os.path.isfile(file_train) or not os.path.isfile(file_test):
        raise FileNotFoundError(
            "Darcy dataset files not found. Set data.auto_download=true to fetch with gdown, "
            "or place files under datasets/<dataset_name>/."
        )

    return file_train, file_test


def flatten_invars(invar_dict):
    parts = []
    for key in sorted(invar_dict.keys()):
        value = np.asarray(invar_dict[key], dtype=np.float32)
        parts.append(value.reshape(value.shape[0], -1))
    return np.concatenate(parts, axis=1)


@hydra.main(version_base="1.3", config_path="conf", config_name="config_DeepO")
def run(cfg: PhysicsNeMoConfig) -> None:
    torch.manual_seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    file_train, file_test = ensure_dataset(cfg)
    invar_train, outvar_train = load_deeponet_dataset(
        file_train,
        ["coeff"],
        ["sol"],
        n_examples=int(cfg.data.n_train),
    )
    invar_test, outvar_test = load_deeponet_dataset(
        file_test,
        ["coeff"],
        ["sol"],
        n_examples=int(cfg.data.n_test),
    )

    x_train = flatten_invars(invar_train)
    y_train = np.asarray(outvar_train["sol"], dtype=np.float32).reshape(-1, 1)
    x_test = flatten_invars(invar_test)
    y_test = np.asarray(outvar_test["sol"], dtype=np.float32).reshape(-1, 1)

    model = FullyConnected(
        in_features=x_train.shape[1],
        out_features=1,
        num_layers=int(cfg.model.num_layers),
        layer_size=int(cfg.model.layer_size),
    ).to(device)

    loss_fn = MSELoss(reduction="mean")
    optimizer = Adam(model.parameters(), lr=float(cfg.optimizer.lr))
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=float(cfg.scheduler.decay_rate))

    x_train_t = torch.from_numpy(x_train).to(device)
    y_train_t = torch.from_numpy(y_train).to(device)
    x_test_t = torch.from_numpy(x_test).to(device)
    y_test_t = torch.from_numpy(y_test).to(device)

    n_train = x_train_t.shape[0]
    n_test = x_test_t.shape[0]

    for step in range(int(cfg.training.max_steps)):
        optimizer.zero_grad()
        idx = torch.randint(0, n_train, (int(cfg.batch_size.train),), device=device)
        pred = model(x_train_t[idx])
        loss = loss_fn(pred, y_train_t[idx])
        loss.backward()
        optimizer.step()
        scheduler.step()

        if step % int(cfg.training.log_every) == 0:
            with torch.no_grad():
                val_idx = torch.randint(
                    0, n_test, (int(cfg.batch_size.validation),), device=device
                )
                val_loss = loss_fn(model(x_test_t[val_idx]), y_test_t[val_idx])
                print(
                    f"step={step:06d} train_mse={loss.item():.4e} "
                    f"val_mse={val_loss.item():.4e}"
                )


if __name__ == "__main__":
    run()
