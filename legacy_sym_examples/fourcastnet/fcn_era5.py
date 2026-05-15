# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Script to train FourcastNet on ERA5 with an explicit v2-style PyTorch loop."""

import itertools
import logging
from pathlib import Path
from typing import Dict, Iterable, Tuple

import hydra
import torch
from omegaconf import DictConfig as PhysicsNeMoConfig
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from hydra.utils import to_absolute_path
from compat.key_compat import Key
from src.dali_dataset import ERA5HDF5GridDaliIterableDataset
from src.dataset import ERA5HDF5GridDataset
from src.fourcastnet import FourcastNetArch

logger = logging.getLogger(__name__)


def _create_dataset(dataset_kind: str, **kwargs):
    valid_dsets = {
        "default": ERA5HDF5GridDataset,
        "dali": ERA5HDF5GridDaliIterableDataset,
    }

    dset_cls = valid_dsets.get(dataset_kind)
    if dset_cls is None:
        raise ValueError(f"Expected one of {list(valid_dsets.keys())}, but got {dataset_kind}")

    logger.info("Dataset: %s", dset_cls.__name__)
    return dset_cls(**kwargs)


def _to_device_dict(tensors: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device=device, dtype=torch.float32) for k, v in tensors.items()}


def _total_relative_l2(pred: Dict[str, torch.Tensor], truth: Dict[str, torch.Tensor]) -> torch.Tensor:
    losses = []
    for key, p in pred.items():
        t = truth[key]
        bsz = p.shape[0]
        pv = p.reshape(bsz, -1)
        tv = t.reshape(bsz, -1)
        diff = torch.linalg.norm(pv - tv, ord=2, dim=1)
        denom = torch.linalg.norm(tv, ord=2, dim=1).clamp_min(1.0e-8)
        losses.append((diff / denom).mean())
    return torch.stack(losses).sum()


def _default_loader(dataset, batch_size: int, num_workers: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        worker_init_fn=getattr(dataset, "worker_init_fn", None),
    )


def _make_iterable(dataset, batch_size: int, num_workers: int, shuffle: bool) -> Iterable[Tuple[Dict, Dict, Dict]]:
    if isinstance(dataset, ERA5HDF5GridDaliIterableDataset):
        return iter(dataset)
    return itertools.cycle(iter(_default_loader(dataset, batch_size, num_workers, shuffle)))


@hydra.main(version_base="1.3", config_path="conf", config_name="config_FCN")
def run(cfg: PhysicsNeMoConfig) -> None:
    torch.manual_seed(int(cfg.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    channels = list(range(int(cfg.custom.n_channels)))
    train_dataset = _create_dataset(
        cfg.custom.train_dataset.kind,
        data_dir=cfg.custom.train_dataset.data_path,
        chans=channels,
        tstep=int(cfg.custom.tstep),
        n_tsteps=int(cfg.custom.n_tsteps),
        patch_size=int(cfg.arch.afno.patch_size),
        batch_size=int(cfg.batch_size.grid),
        num_workers=int(cfg.custom.num_workers.grid),
        shuffle=True,
    )

    test_dataset = _create_dataset(
        cfg.custom.test_dataset.kind,
        data_dir=cfg.custom.test_dataset.data_path,
        chans=channels,
        tstep=int(cfg.custom.tstep),
        n_tsteps=int(cfg.custom.n_tsteps),
        patch_size=int(cfg.arch.afno.patch_size),
        n_samples_per_year=int(cfg.custom.test_dataset.n_samples_per_year),
        batch_size=int(cfg.batch_size.validation),
        num_workers=int(cfg.custom.num_workers.validation),
        shuffle=False,
    )

    input_keys = [Key(k, size=train_dataset.nchans) for k in train_dataset.invar_keys]
    output_keys = [Key(k, size=train_dataset.nchans) for k in train_dataset.outvar_keys]

    model = FourcastNetArch(
        input_keys=input_keys,
        output_keys=output_keys,
        img_shape=test_dataset.img_shape,
        patch_size=int(cfg.arch.afno.patch_size),
        embed_dim=int(cfg.arch.afno.embed_dim),
        depth=int(cfg.arch.afno.depth),
        num_blocks=int(cfg.arch.afno.num_blocks),
    ).to(device)

    optimizer = Adam(model.parameters(), lr=float(cfg.optimizer.lr))
    scheduler = CosineAnnealingLR(optimizer, T_max=int(cfg.scheduler.T_max))

    amp_enabled = bool(cfg.training.amp) and torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    train_iter = _make_iterable(
        train_dataset,
        int(cfg.batch_size.grid),
        int(cfg.custom.num_workers.grid),
        shuffle=True,
    )
    val_iter = _make_iterable(
        test_dataset,
        int(cfg.batch_size.validation),
        int(cfg.custom.num_workers.validation),
        shuffle=False,
    )

    max_steps = int(cfg.training.max_steps)
    print_freq = int(cfg.training.print_stats_freq)
    val_freq = int(cfg.training.validation_freq)
    save_freq = int(cfg.training.save_network_freq)
    ckpt_name = str(cfg.training.checkpoint_name)

    model.train()
    for step in range(max_steps):
        invar, outvar, _ = next(train_iter)
        invar = _to_device_dict(invar, device)
        outvar = _to_device_dict(outvar, device)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            pred = model(invar)
            loss = _total_relative_l2(pred, outvar)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        if step % print_freq == 0:
            lr = scheduler.get_last_lr()[0]
            logger.info("step=%06d train_loss=%.4e lr=%.4e", step, loss.item(), lr)

        if val_freq > 0 and step % val_freq == 0:
            model.eval()
            with torch.no_grad():
                vin, vout, _ = next(val_iter)
                vin = _to_device_dict(vin, device)
                vout = _to_device_dict(vout, device)
                vpred = model(vin)
                vloss = _total_relative_l2(vpred, vout)
                logger.info("step=%06d val_loss=%.4e", step, vloss.item())
            model.train()

        if save_freq > 0 and step > 0 and step % save_freq == 0:
            torch.save(model.state_dict(), ckpt_name)
            logger.info("checkpoint saved: %s", ckpt_name)

    torch.save(model.state_dict(), ckpt_name)
    logger.info("training complete, final checkpoint: %s", ckpt_name)


if __name__ == "__main__":
    run()
