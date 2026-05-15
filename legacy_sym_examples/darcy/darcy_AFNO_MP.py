# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

import hydra
import numpy as np
import torch
import torch.distributed as dist
from omegaconf import DictConfig as PhysicsNeMoConfig
from torch.nn import MSELoss
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler

from physicsnemo.models.afno import AFNO

from utilities import download_FNO_dataset, load_FNO_dataset


def init_dist() -> tuple[int, int, int, bool]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    is_dist = world_size > 1

    if is_dist and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)

    return rank, local_rank, world_size, is_dist


def cleanup_dist(is_dist: bool) -> None:
    if is_dist and dist.is_initialized():
        dist.destroy_process_group()


def ensure_dataset(cfg: PhysicsNeMoConfig, rank: int, is_dist: bool) -> tuple[str, str]:
    dataset_name = str(cfg.data.dataset_name)
    dataset_dir = str(cfg.data.dataset_dir)
    file_train = os.path.join(dataset_dir, dataset_name, "piececonst_r241_N1024_smooth1.hdf5")
    file_test = os.path.join(dataset_dir, dataset_name, "piececonst_r241_N1024_smooth2.hdf5")

    if bool(cfg.data.auto_download) and rank == 0:
        download_FNO_dataset(dataset_name, outdir=dataset_dir)

    if is_dist:
        dist.barrier()

    if not os.path.isfile(file_train) or not os.path.isfile(file_test):
        raise FileNotFoundError(
            "Darcy dataset files not found. Set data.auto_download=true to fetch with gdown, "
            "or place files under datasets/<dataset_name>/."
        )

    return file_train, file_test


@hydra.main(version_base="1.3", config_path="conf", config_name="config_AFNO_MP")
def run(cfg: PhysicsNeMoConfig) -> None:
    rank, local_rank, _, is_dist = init_dist()

    torch.manual_seed(int(cfg.seed) + rank)
    np.random.seed(int(cfg.seed) + rank)

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    file_train, file_test = ensure_dataset(cfg, rank, is_dist)

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

    img_shape = [
        next(iter(invar_train.values())).shape[-2],
        next(iter(invar_train.values())).shape[-1],
    ]
    img_shape = [s - s % int(cfg.model.patch_size) for s in img_shape]

    for d in (invar_train, outvar_train, invar_test, outvar_test):
        for k in d:
            d[k] = d[k][:, :, : img_shape[0], : img_shape[1]]

    x_train = torch.from_numpy(invar_train["coeff"].astype(np.float32))
    y_train = torch.from_numpy(outvar_train["sol"].astype(np.float32))
    x_test = torch.from_numpy(invar_test["coeff"].astype(np.float32)).to(device)
    y_test = torch.from_numpy(outvar_test["sol"].astype(np.float32)).to(device)

    train_dataset = TensorDataset(x_train, y_train)
    train_sampler = DistributedSampler(train_dataset, shuffle=True) if is_dist else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(cfg.batch_size.train),
        shuffle=train_sampler is None,
        sampler=train_sampler,
        drop_last=True,
    )

    model = AFNO(
        inp_shape=img_shape,
        in_channels=int(cfg.model.in_channels),
        out_channels=int(cfg.model.out_channels),
        patch_size=[int(cfg.model.patch_size), int(cfg.model.patch_size)],
        embed_dim=int(cfg.model.embed_dim),
        depth=int(cfg.model.depth),
        num_blocks=int(cfg.model.num_blocks),
    ).to(device)

    if is_dist:
        model = DDP(model, device_ids=[local_rank] if torch.cuda.is_available() else None)

    loss_fn = MSELoss(reduction="mean")
    optimizer = Adam(model.parameters(), lr=float(cfg.optimizer.lr))
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=float(cfg.scheduler.decay_rate))

    iterator = iter(train_loader)
    for step in range(int(cfg.training.max_steps)):
        if is_dist and train_sampler is not None and step % len(train_loader) == 0:
            train_sampler.set_epoch(step // len(train_loader))

        try:
            xb, yb = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            xb, yb = next(iterator)

        xb = xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        optimizer.step()
        scheduler.step()

        if step % int(cfg.training.log_every) == 0 and rank == 0:
            with torch.no_grad():
                val_idx = torch.randint(
                    0,
                    x_test.shape[0],
                    (int(cfg.batch_size.validation),),
                    device=device,
                )
                val_loss = loss_fn(model(x_test[val_idx]), y_test[val_idx])
                print(
                    f"step={step:06d} train_mse={loss.item():.4e} "
                    f"val_mse={val_loss.item():.4e}"
                )

    cleanup_dist(is_dist)


if __name__ == "__main__":
    run()
