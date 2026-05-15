# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FourcastNet AFNO wrapper without legacy framework dependencies."""

import logging
from typing import Dict, List, Tuple

import torch
from torch import Tensor, nn

from physicsnemo.models.afno import AFNO
from compat.key_compat import Key


class FourcastNetArch(nn.Module):
    """Autoregressive wrapper around AFNO for multi-step prediction."""

    def __init__(
        self,
        input_keys: List[Key],
        output_keys: List[Key],
        img_shape: Tuple[int, int],
        detach_keys: List[Key] = None,
        patch_size: int = 16,
        embed_dim: int = 256,
        depth: int = 4,
        num_blocks: int = 4,
    ) -> None:
        super().__init__()
        self.input_keys = list(input_keys)
        self.output_keys = list(output_keys)
        self.detach_keys = set(k.name for k in (detach_keys or []))

        if len(self.input_keys) != 1:
            raise ValueError("FourcastNet accepts exactly one input variable (x_t0)")

        self.n_tsteps = len(self.output_keys)
        logging.info("Unrolling FourcastNet over %d timesteps", self.n_tsteps)

        in_channels = int(self.input_keys[0].size)
        out_channels = int(self.output_keys[0].size)

        self._impl = AFNO(
            inp_shape=[int(img_shape[0]), int(img_shape[1])],
            in_channels=in_channels,
            out_channels=out_channels,
            patch_size=[int(patch_size), int(patch_size)],
            embed_dim=int(embed_dim),
            depth=int(depth),
            num_blocks=int(num_blocks),
        )

    def make_node(self, name: str = "FCN"):
        raise RuntimeError(
            "make_node is not supported in the v2-style wrapper. "
            f"Use the module directly: {name}."
        )

    def forward(self, in_vars: Dict[str, Tensor]) -> Dict[str, Tensor]:
        in_name = self.input_keys[0].name
        if in_name not in in_vars:
            raise KeyError(f"Missing required input key: {in_name}")

        x = in_vars[in_name]
        if in_name in self.detach_keys:
            x = x.detach()

        out: Dict[str, Tensor] = {}
        for t, out_key in enumerate(self.output_keys):
            x = self._impl(x)
            if out_key.name in self.detach_keys:
                x = x.detach()
            out[out_key.name] = x
        return out
