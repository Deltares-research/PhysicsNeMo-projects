# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Dict

import torch
from torch import nn


class Aggregator(nn.Module):
    """Minimal compatibility layer for legacy custom loss aggregators."""

    def __init__(self, params, num_losses, weights=None):
        super().__init__()
        _ = params
        _ = num_losses
        self.weights = weights or {}
        self.register_buffer("init_loss", torch.tensor(0.0, dtype=torch.float32))

    @staticmethod
    def weigh_losses(losses: Dict[str, torch.Tensor], weights: Dict[str, float]):
        if not weights:
            return losses
        out = {}
        for key, value in losses.items():
            out[key] = value * float(weights.get(key, 1.0))
        return out
