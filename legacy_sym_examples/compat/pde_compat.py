# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0


class PDE:
    """Minimal compatibility base for legacy symbolic PDE definition classes."""

    name = "PDE"

    def __init__(self):
        self.equations = {}
