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
from physicsnemo.models.afno import AFNO

from utilities import download_FNO_dataset, load_FNO_dataset


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


@hydra.main(version_base="1.3", config_path="conf", config_name="config_AFNO")
def run(cfg: PhysicsNeMoConfig) -> None:
    torch.manual_seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    file_train, file_test = ensure_dataset(cfg)
    invar_train, outvar_train = load_FNO_dataset(
        file_train,
        ["coeff"],
        ["sol"],
        n_examples=int(cfg.data.n_train),
    )
    invar_test, outvar_test = load_FNO_dataset(
        file_test,
        ["coeff"],
        ["sol"],
        n_examples=int(cfg.data.n_test),
    )

    # get training image shape
    img_shape = [
        next(iter(invar_train.values())).shape[-2],
        next(iter(invar_train.values())).shape[-1],
    ]

    # crop out some pixels so that img_shape is divisible by patch_size of AFNO
    img_shape = [s - s % int(cfg.model.patch_size) for s in img_shape]
    print(f"cropped img_shape: {img_shape}")
    for d in (invar_train, outvar_train, invar_test, outvar_test):
        for k in d:
            d[k] = d[k][:, :, : img_shape[0], : img_shape[1]]
            print(f"{k}: {d[k].shape}")

    x_train = torch.from_numpy(invar_train["coeff"].astype(np.float32)).to(device)
    y_train = torch.from_numpy(outvar_train["sol"].astype(np.float32)).to(device)
    x_test = torch.from_numpy(invar_test["coeff"].astype(np.float32)).to(device)
    y_test = torch.from_numpy(outvar_test["sol"].astype(np.float32)).to(device)

    model = AFNO(
        inp_shape=img_shape,
        in_channels=int(cfg.model.in_channels),
        out_channels=int(cfg.model.out_channels),
        patch_size=[int(cfg.model.patch_size), int(cfg.model.patch_size)],
        embed_dim=int(cfg.model.embed_dim),
        depth=int(cfg.model.depth),
        num_blocks=int(cfg.model.num_blocks),
    ).to(device)

    loss_fn = MSELoss(reduction="mean")
    optimizer = Adam(model.parameters(), lr=float(cfg.optimizer.lr))
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=float(cfg.scheduler.decay_rate))

    n_train = x_train.shape[0]
    n_test = x_test.shape[0]

    for step in range(int(cfg.training.max_steps)):
        optimizer.zero_grad()
        idx = torch.randint(0, n_train, (int(cfg.batch_size.train),), device=device)
        loss = loss_fn(model(x_train[idx]), y_train[idx])
        loss.backward()
        optimizer.step()
        scheduler.step()

        if step % int(cfg.training.log_every) == 0:
            with torch.no_grad():
                val_idx = torch.randint(
                    0, n_test, (int(cfg.batch_size.validation),), device=device
                )
                val_loss = loss_fn(model(x_test[val_idx]), y_test[val_idx])
                print(
                    f"step={step:06d} train_mse={loss.item():.4e} "
                    f"val_mse={val_loss.item():.4e}"
                )


if __name__ == "__main__":
    run()
