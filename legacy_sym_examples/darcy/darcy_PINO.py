# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

import hydra
import numpy as np
import torch
from omegaconf import DictConfig as PhysicsNeMoConfig
from torch.nn import MSELoss
from torch.optim import Adam, lr_scheduler

from physicsnemo.models.fno import FNO
from ops import ddx, dx
from utilities import download_FNO_dataset, load_FNO_dataset


def darcy_residual(sol: torch.Tensor, coeff: torch.Tensor) -> torch.Tensor:
    dxf = 1.0 / sol.shape[-2]
    dyf = 1.0 / sol.shape[-1]

    dcdx = dx(coeff, dx=dxf, channel=0, dim=0, order=1, padding="replication")
    dcdy = dx(coeff, dx=dyf, channel=0, dim=1, order=1, padding="replication")

    dudx = dx(sol, dx=dxf, channel=0, dim=0, order=1, padding="replication")
    dudy = dx(sol, dx=dyf, channel=0, dim=1, order=1, padding="replication")
    dduddx = ddx(sol, dx=dxf, channel=0, dim=0, order=1, padding="replication")
    dduddy = ddx(sol, dx=dyf, channel=0, dim=1, order=1, padding="replication")

    res = 1.0 + (dcdx * dudx) + (coeff * dduddx) + (dcdy * dudy) + (coeff * dduddy)
    return torch.nn.functional.pad(res[:, :, 2:-2, 2:-2], [2, 2, 2, 2], "constant", 0.0)


@hydra.main(version_base="1.3", config_path="conf", config_name="config_PINO")
def run(cfg: PhysicsNeMoConfig) -> None:
    torch.manual_seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if bool(cfg.data.auto_download):
        download_FNO_dataset(str(cfg.data.dataset_name), outdir=str(cfg.data.dataset_dir))

    train_path = f"{cfg.data.dataset_dir}/Darcy_241/piececonst_r241_N1024_smooth1.hdf5"
    test_path = f"{cfg.data.dataset_dir}/Darcy_241/piececonst_r241_N1024_smooth2.hdf5"

    invar_train, outvar_train = load_FNO_dataset(
        train_path,
        ["coeff"],
        ["sol"],
        n_examples=int(cfg.data.n_train),
    )
    invar_test, outvar_test = load_FNO_dataset(
        test_path,
        ["coeff"],
        ["sol"],
        n_examples=int(cfg.data.n_test),
    )

    x_train = torch.from_numpy(invar_train["coeff"].astype(np.float32)).to(device)
    y_train = torch.from_numpy(outvar_train["sol"].astype(np.float32)).to(device)
    x_test = torch.from_numpy(invar_test["coeff"].astype(np.float32)).to(device)
    y_test = torch.from_numpy(outvar_test["sol"].astype(np.float32)).to(device)

    model = FNO(
        in_channels=int(cfg.model.in_channels),
        out_channels=int(cfg.model.out_channels),
        decoder_layers=int(cfg.model.decoder_layers),
        decoder_layer_size=int(cfg.model.decoder_layer_size),
        dimension=int(cfg.model.dimension),
        latent_channels=int(cfg.model.latent_channels),
        num_fno_layers=int(cfg.model.num_fno_layers),
        num_fno_modes=int(cfg.model.num_fno_modes),
        padding=int(cfg.model.padding),
    ).to(device)

    mse = MSELoss(reduction="mean")
    optimizer = Adam(model.parameters(), lr=float(cfg.optimizer.lr))
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=float(cfg.scheduler.decay_rate))

    n_train = x_train.shape[0]
    n_test = x_test.shape[0]
    w_sol = float(cfg.loss.weights.sol)
    w_darcy = float(cfg.loss.weights.darcy)

    for step in range(int(cfg.training.max_steps)):
        optimizer.zero_grad()
        idx = torch.randint(0, n_train, (int(cfg.batch_size.train),), device=device)
        xb = x_train[idx]
        yb = y_train[idx]

        pred = model(xb)
        loss_sol = mse(pred, yb)
        res = darcy_residual(pred, xb)
        loss_darcy = mse(res, torch.zeros_like(res))
        loss = w_sol * loss_sol + w_darcy * loss_darcy

        loss.backward()
        optimizer.step()
        scheduler.step()

        if step % int(cfg.training.log_every) == 0:
            with torch.no_grad():
                vidx = torch.randint(0, n_test, (int(cfg.batch_size.validation),), device=device)
                vpred = model(x_test[vidx])
                vsol = mse(vpred, y_test[vidx])
                vres = darcy_residual(vpred, x_test[vidx])
                vdarcy = mse(vres, torch.zeros_like(vres))
                print(
                    f"step={step:06d} train_sol={loss_sol.item():.4e} train_darcy={loss_darcy.item():.4e} "
                    f"val_sol={vsol.item():.4e} val_darcy={vdarcy.item():.4e}"
                )


if __name__ == "__main__":
    run()
