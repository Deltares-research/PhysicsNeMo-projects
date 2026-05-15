# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass


@dataclass(frozen=True)
class Key:
    """Compatibility shim for legacy examples that expected the old Key API."""

    name: str
    size: int = 1
    scale: float = 1.0

    def __hash__(self):
        return hash((self.name, self.size, self.scale))
