# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hydra
import numpy as np
import torch
from omegaconf import DictConfig as PhysicsNeMoConfig
from scipy import optimize
from torch import nn
from torch.optim import Adam, lr_scheduler


class LookupNet(nn.Module):
    def __init__(self, in_features: int, out_features: int, hidden_dim: int, num_layers: int):
        super().__init__()
        layers = [nn.Linear(in_features, hidden_dim), nn.Tanh()]
        for _ in range(max(0, num_layers - 1)):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
        layers.append(nn.Linear(hidden_dim, out_features))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def compute_u_tau(U: np.ndarray, Y: np.ndarray, nu: float) -> np.ndarray:
    def f(u_tau, y, u):
        return u_tau * np.log(9.793 * y * u_tau / nu) - u * 0.4187

    def fprime(u_tau, y, u):
        return 1 + np.log(9.793 * y * u_tau / nu)

    out = []
    for i in range(len(U)):
        out.append(
            optimize.newton(
                f,
                1.0,
                fprime=fprime,
                args=(Y[i], U[i]),
                tol=1.48e-08,
                maxiter=200,
            )
        )
    return np.asarray(out, dtype=np.float32)


def build_train_data() -> tuple[np.ndarray, np.ndarray]:
    u = np.linspace(1e-3, 50, num=100, dtype=np.float32)
    y = np.linspace(1e-3, 0.5, num=100, dtype=np.float32)
    U, Y = np.meshgrid(u, y)
    U = U.reshape(-1)
    Y = Y.reshape(-1)

    nu = 1.0 / 590.0
    u_tau = compute_u_tau(U, Y, nu)

    train = np.concatenate(
        [U.reshape(-1, 1), Y.reshape(-1, 1), u_tau.reshape(-1, 1)],
        axis=1,
    )
    np.savetxt("u_tau.csv", train, delimiter=",")

    invar = np.concatenate([U.reshape(-1, 1), Y.reshape(-1, 1)], axis=1)
    outvar = u_tau.reshape(-1, 1)
    return invar, outvar


def build_val_data() -> tuple[np.ndarray, np.ndarray]:
    u = np.random.uniform(1e-3, 50, size=100).astype(np.float32)
    y = np.random.uniform(1e-3, 0.5, size=100).astype(np.float32)
    U, Y = np.meshgrid(u, y)
    U = U.reshape(-1)
    Y = Y.reshape(-1)

    nu = 1.0 / 590.0
    u_tau = compute_u_tau(U, Y, nu)

    val = np.concatenate(
        [U.reshape(-1, 1), Y.reshape(-1, 1), u_tau.reshape(-1, 1)],
        axis=1,
    )
    np.savetxt("u_tau_val.csv", val, delimiter=",")

    invar = np.concatenate([U.reshape(-1, 1), Y.reshape(-1, 1)], axis=1)
    outvar = u_tau.reshape(-1, 1)
    return invar, outvar


@hydra.main(version_base="1.3", config_path="conf_u_tau_lookup", config_name="config")
def run(cfg: PhysicsNeMoConfig) -> None:
    torch.manual_seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x_train_np, y_train_np = build_train_data()
    x_val_np, y_val_np = build_val_data()

    x_train = torch.from_numpy(x_train_np.astype(np.float32)).to(device)
    y_train = torch.from_numpy(y_train_np.astype(np.float32)).to(device)
    x_val = torch.from_numpy(x_val_np.astype(np.float32)).to(device)
    y_val = torch.from_numpy(y_val_np.astype(np.float32)).to(device)

    model = LookupNet(
        in_features=2,
        out_features=1,
        hidden_dim=int(cfg.model.hidden_dim),
        num_layers=int(cfg.model.num_layers),
    ).to(device)

    optimizer = Adam(model.parameters(), lr=float(cfg.optimizer.lr))
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=float(cfg.scheduler.decay_rate))
    loss_fn = nn.MSELoss(reduction="mean")

    n_train = x_train.shape[0]
    for step in range(int(cfg.training.max_steps)):
        optimizer.zero_grad()

        idx = torch.randint(0, n_train, (int(cfg.batch_size.train),), device=device)
        pred = model(x_train[idx])
        loss = loss_fn(pred, y_train[idx])

        loss.backward()
        optimizer.step()
        scheduler.step()

        if step % int(cfg.training.log_every) == 0:
            with torch.no_grad():
                val_pred = model(x_val)
                val_loss = loss_fn(val_pred, y_val)
                print(
                    f"step={step:06d} train_mse={loss.item():.4e} "
                    f"val_mse={val_loss.item():.4e}"
                )


if __name__ == "__main__":
    run()
