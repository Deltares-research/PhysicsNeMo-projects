# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

import hydra
import numpy as np
import torch
from omegaconf import DictConfig as PhysicsNeMoConfig
from torch import nn
from torch.optim import Adam, lr_scheduler


class SRNet(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, scale: int):
        super().__init__()
        self.pre = nn.Sequential(
            nn.Conv3d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv3d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.scale = scale
        self.post = nn.Sequential(
            nn.Conv3d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv3d(hidden_channels, in_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor):
        h = self.pre(x)
        h = nn.functional.interpolate(h, scale_factor=self.scale, mode="trilinear", align_corners=False)
        return self.post(h)


def make_synthetic_batch(batch: int, low_size: int, scale: int, device):
    low = torch.randn(batch, 3, low_size, low_size, low_size, device=device)
    high = nn.functional.interpolate(low, scale_factor=scale, mode="trilinear", align_corners=False)
    high = high + 0.02 * torch.randn_like(high)
    return low, high


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def run(cfg: PhysicsNeMoConfig) -> None:
    torch.manual_seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    scale = int(cfg.model.scaling_factor)
    low_size = int(cfg.data.low_res_size)
    model = SRNet(in_channels=3, hidden_channels=int(cfg.model.hidden_channels), scale=scale).to(device)

    optimizer = Adam(model.parameters(), lr=float(cfg.optimizer.lr))
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=float(cfg.scheduler.decay_rate))
    mse = nn.MSELoss()

    for step in range(int(cfg.training.max_steps)):
        optimizer.zero_grad()
        x_lr, y_hr = make_synthetic_batch(int(cfg.batch_size.train), low_size, scale, device)
        pred = model(x_lr)
        loss_u = mse(pred, y_hr)

        pred_dx = pred[:, :, 1:, :, :] - pred[:, :, :-1, :, :]
        y_dx = y_hr[:, :, 1:, :, :] - y_hr[:, :, :-1, :, :]
        loss_grad = mse(pred_dx, y_dx)

        loss = float(cfg.loss.U) * loss_u + float(cfg.loss.dU) * loss_grad
        loss.backward()
        optimizer.step()
        scheduler.step()

        if step % int(cfg.training.log_every) == 0:
            print(f"step={step:06d} total={loss.item():.4e} U={loss_u.item():.4e} dU={loss_grad.item():.4e}")


if __name__ == "__main__":
    run()
