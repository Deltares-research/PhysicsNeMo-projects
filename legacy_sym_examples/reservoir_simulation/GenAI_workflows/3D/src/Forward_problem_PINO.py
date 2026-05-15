# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

import argparse
from pathlib import Path

from omegaconf import OmegaConf

from compat_runner import run_training_from_cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", type=str, default="config_PINO")
    args, unknown = parser.parse_known_args()

    conf_path = Path(__file__).resolve().parent / "conf" / f"{args.config_name}.yaml"
    cfg = OmegaConf.load(str(conf_path))
    if unknown:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(unknown))

    run_training_from_cfg(cfg, default_dim=3, workflow_name="Forward_problem_PINO", scenario_name="GENAI_3D")


if __name__ == "__main__":
    main()
