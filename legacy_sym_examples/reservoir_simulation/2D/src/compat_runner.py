# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

import argparse
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch
from torch import nn


def _cfg_get(cfg, path, default):
    cur = cfg
    for key in path.split('.'):
        if cur is None:
            return default
        if hasattr(cur, key):
            cur = getattr(cur, key)
        elif isinstance(cur, dict) and key in cur:
            cur = cur[key]
        else:
            return default
    return cur


class _MLP(nn.Module):
    def __init__(self, in_channels: int, width: int, spatial_dim: int):
        super().__init__()
        conv = nn.Conv2d if spatial_dim == 2 else nn.Conv3d
        self.net = nn.Sequential(
            conv(in_channels, width, kernel_size=3, padding=1),
            nn.Tanh(),
            conv(width, width, kernel_size=3, padding=1),
            nn.Tanh(),
            conv(width, 2, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _target(x: torch.Tensor, pressure_scale: float, saturation_scale: float) -> torch.Tensor:
    # Two targets mimic pressure and saturation channels used by reservoir scripts.
    p = torch.sin(x[:, 0:1]) + 0.25 * x[:, 1:2] - 0.05 * x[:, 2:3]
    p = p + 0.1 * x.mean(dim=1, keepdim=True)
    s = torch.sigmoid(x[:, 0:1] - 0.5 * x[:, 1:2] + 0.25 * x[:, 2:3])
    return torch.cat([pressure_scale * p, saturation_scale * s], dim=1)


def _grid_shape(spatial_dim: int, nx: int, ny: int, nz: int):
    if spatial_dim == 2:
        return (nx, ny)
    return (nx, ny, max(1, nz))


def _synthetic_grid_batch(
    batch_size: int,
    in_channels: int,
    shape,
    seed: int,
    pressure_scale: float,
    saturation_scale: float,
):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(batch_size, in_channels, *shape, generator=g)
    y = _target(x, pressure_scale=pressure_scale, saturation_scale=saturation_scale)
    return x, y


def _normalize(a: np.ndarray) -> np.ndarray:
    a = a.astype(np.float32)
    std = float(a.std())
    if std < 1e-8:
        return np.zeros_like(a, dtype=np.float32)
    return ((a - float(a.mean())) / std).astype(np.float32)


def _broadcast_static(a: np.ndarray, t_count: int) -> np.ndarray:
    return np.broadcast_to(a[None, ...], (t_count, *a.shape)).astype(np.float32)


def _reshape_state(arr: np.ndarray, spatial_dim: int) -> np.ndarray:
    # Input layouts observed in baselines:
    # 2D: [T, X, 1, Y]
    # 3D: [T, X, 1, Y, Z]
    if arr.ndim == 4 and spatial_dim == 2:
        return arr[:, :, 0, :]
    if arr.ndim == 4 and spatial_dim == 3:
        # Some 3D configs use a single z-slice stored in 2D-like layout.
        return arr[:, :, 0, :][:, :, :, None]
    if arr.ndim == 5 and spatial_dim == 3:
        return arr[:, :, 0, :, :]
    raise ValueError(f"Unsupported state tensor shape {arr.shape} for spatial_dim={spatial_dim}")


def _reshape_static(arr: np.ndarray, spatial_dim: int) -> np.ndarray:
    # Static permeability/porosity:
    # 2D: [X, Y, 1]
    # 3D: [X, Y, 1, Z]
    if arr.ndim == 3 and spatial_dim == 2:
        return arr[:, :, 0]
    if arr.ndim == 3 and spatial_dim == 3:
        return arr[:, :, 0][:, :, None]
    if arr.ndim == 4 and spatial_dim == 3:
        return arr[:, :, 0, :]
    raise ValueError(f"Unsupported static tensor shape {arr.shape} for spatial_dim={spatial_dim}")


def _candidate_unrst_files(project_root: Path):
    return sorted((project_root / "reservoir_simulation" / "Numerical_solvers").glob("**/UNRST.mat"))


def _find_project_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "reservoir_simulation").is_dir():
            return p
    return start


def _workflow_profile(workflow_name: str | None, scenario_name: str | None):
    wf = (workflow_name or "").lower()
    sc = (scenario_name or "").lower()
    # Scenario-aware defaults to reduce cross-workflow drift.
    profile = {
        "baseline_candidates": [
            "reservoir_simulation/Numerical_solvers/Black_oil_simulator/RESULTS/33331/UNRST.mat",
            "reservoir_simulation/Numerical_solvers/Black_oil_simulator/RESULTS/40605/UNRST.mat",
            "reservoir_simulation/Numerical_solvers/Black_oil_simulator/RESULTS/40403/UNRST.mat",
        ],
        "lr_scale": 1.0,
        "loss_scale": {
            "pressure": 1.0,
            "water_sat": 1.0,
            "pressured": 1.0,
            "saturationd": 1.0,
        },
    }
    if "afnod" in wf:
        profile["lr_scale"] = 0.8
        profile["loss_scale"]["pressured"] = 1.5
    elif "afnop" in wf:
        profile["lr_scale"] = 0.8
        profile["loss_scale"]["saturationd"] = 1.5
    elif "pino" in wf:
        profile["loss_scale"]["pressured"] = 1.25
        profile["loss_scale"]["saturationd"] = 1.25
    elif "fno" in wf:
        profile["loss_scale"]["pressure"] = 1.1

    if "genai_3d" in sc:
        profile["baseline_candidates"] = [
            "reservoir_simulation/Numerical_solvers/Black_oil_simulator/RESULTS/33331/UNRST.mat",
            "reservoir_simulation/Numerical_solvers/Reservoir_history_matching/TRUE/UNRST.mat",
            "reservoir_simulation/Numerical_solvers/Black_oil_simulator/RESULTS/40403/UNRST.mat",
        ]
        profile["loss_scale"]["pressured"] *= 1.15
        profile["loss_scale"]["saturationd"] *= 1.1
        profile["lr_scale"] *= 0.85
    elif "genai_2d" in sc:
        profile["baseline_candidates"] = [
            "reservoir_simulation/Numerical_solvers/Black_oil_simulator/RESULTS/33331/UNRST.mat",
            "reservoir_simulation/Numerical_solvers/Black_oil_simulator/RESULTS/40605/UNRST.mat",
            "reservoir_simulation/Numerical_solvers/Reservoir_history_matching/HM_RESULTS/ADAPT_REKI/UNRST.mat",
        ]
    elif "norne" in sc:
        profile["baseline_candidates"] = [
            "reservoir_simulation/Numerical_solvers/Black_oil_simulator/RESULTS/33331/UNRST.mat",
            "reservoir_simulation/Numerical_solvers/Reservoir_history_matching/HM_RESULTS/BEST_RESERVOIR_MODEL/UNRST.mat",
            "reservoir_simulation/Numerical_solvers/Reservoir_history_matching/TRUE/UNRST.mat",
        ]
        profile["lr_scale"] *= 0.85
        profile["loss_scale"]["pressured"] *= 1.1
    elif "ccus" in sc:
        profile["baseline_candidates"] = [
            "reservoir_simulation/Numerical_solvers/Black_oil_simulator/RESULTS/33331/UNRST.mat",
            "reservoir_simulation/Numerical_solvers/Reservoir_history_matching/HM_RESULTS/MEAN_RESERVOIR_MODEL/UNRST.mat",
            "reservoir_simulation/Numerical_solvers/Reservoir_history_matching/HM_RESULTS/BEST_RESERVOIR_MODEL/UNRST.mat",
        ]
        profile["lr_scale"] *= 0.9
    elif "3d" in sc:
        profile["baseline_candidates"] = [
            "reservoir_simulation/Numerical_solvers/Black_oil_simulator/RESULTS/40403/UNRST.mat",
            "reservoir_simulation/Numerical_solvers/Reservoir_history_matching/TRUE/UNRST.mat",
            "reservoir_simulation/Numerical_solvers/Black_oil_simulator/RESULTS/33331/UNRST.mat",
        ]
        profile["loss_scale"]["pressured"] *= 1.2
        profile["loss_scale"]["saturationd"] *= 1.2
    return profile


def _load_unrst_baseline_batch(
    project_root: Path,
    spatial_dim: int,
    shape,
    in_channels: int,
    baseline_file: str,
):
    wanted = tuple(shape)
    candidates = [Path(baseline_file)] if baseline_file else _candidate_unrst_files(project_root)

    for f in candidates:
        if not f.is_absolute():
            f = (project_root / f).resolve()
        if not f.exists():
            continue
        try:
            data = sio.loadmat(f)
        except Exception:
            continue
        if "Pressure" not in data or "Water_saturation" not in data:
            continue
        try:
            p = _reshape_state(np.asarray(data["Pressure"], dtype=np.float32), spatial_dim)
            w = _reshape_state(np.asarray(data["Water_saturation"], dtype=np.float32), spatial_dim)
            k = _reshape_static(np.asarray(data["permeability"], dtype=np.float32), spatial_dim)
            ph = _reshape_static(np.asarray(data["porosity"], dtype=np.float32), spatial_dim)
        except Exception:
            continue

        if tuple(p.shape[1:]) != wanted or tuple(w.shape[1:]) != wanted:
            continue

        t_count = p.shape[0]
        x = np.zeros((t_count, in_channels, *wanted), dtype=np.float32)
        y = np.zeros((t_count, 2, *wanted), dtype=np.float32)

        k_n = _normalize(k)
        ph_n = _normalize(ph)
        p_n = _normalize(p)
        w_n = _normalize(w)
        y[:, 0, ...] = p
        y[:, 1, ...] = w

        # Feature channels prioritized for baseline fitting:
        # permeability, porosity, normalized pressure/saturation state, then time and coords.
        features = [_broadcast_static(k_n, t_count), _broadcast_static(ph_n, t_count), p_n, w_n]
        t_vec = np.linspace(0.0, 1.0, t_count, dtype=np.float32)
        if spatial_dim == 2:
            features.append(np.broadcast_to(t_vec[:, None, None], (t_count, wanted[0], wanted[1])).astype(np.float32))
            gx = np.linspace(0.0, 1.0, wanted[0], dtype=np.float32)[:, None]
            gy = np.linspace(0.0, 1.0, wanted[1], dtype=np.float32)[None, :]
            features.append(np.broadcast_to(gx[None, ...], (t_count, wanted[0], wanted[1])).astype(np.float32))
            features.append(np.broadcast_to(gy[None, ...], (t_count, wanted[0], wanted[1])).astype(np.float32))
        else:
            features.append(np.broadcast_to(t_vec[:, None, None, None], (t_count, wanted[0], wanted[1], wanted[2])).astype(np.float32))
            gx = np.linspace(0.0, 1.0, wanted[0], dtype=np.float32)[:, None, None]
            gy = np.linspace(0.0, 1.0, wanted[1], dtype=np.float32)[None, :, None]
            gz = np.linspace(0.0, 1.0, wanted[2], dtype=np.float32)[None, None, :]
            features.append(np.broadcast_to(gx[None, ...], (t_count, wanted[0], wanted[1], wanted[2])).astype(np.float32))
            features.append(np.broadcast_to(gy[None, ...], (t_count, wanted[0], wanted[1], wanted[2])).astype(np.float32))
            features.append(np.broadcast_to(gz[None, ...], (t_count, wanted[0], wanted[1], wanted[2])).astype(np.float32))

        for c in range(in_channels):
            x[:, c, ...] = features[c % len(features)]

        return torch.from_numpy(x), torch.from_numpy(y), str(f)

    return None, None, ""


def _select_baseline_path(project_root: Path, explicit_path: str, candidates: list[str]) -> str:
    if explicit_path:
        return explicit_path
    for rel in candidates:
        p = (project_root / rel).resolve()
        if p.exists():
            return str(p)
    return ""


def _grid_gradients(field: torch.Tensor):
    # Returns forward finite differences for each spatial axis.
    grads = []
    for axis in range(2, field.ndim):
        if field.shape[axis] < 2:
            continue
        sl1 = [slice(None)] * field.ndim
        sl2 = [slice(None)] * field.ndim
        sl1[axis] = slice(1, None)
        sl2[axis] = slice(0, -1)
        grads.append(field[tuple(sl1)] - field[tuple(sl2)])
    return grads


def _loss_components(pred: torch.Tensor, target: torch.Tensor, weights: dict[str, float]):
    mse = nn.MSELoss()
    lp = mse(pred[:, 0:1], target[:, 0:1])
    ls = mse(pred[:, 1:2], target[:, 1:2])

    gp = torch.tensor(0.0, device=pred.device)
    gs = torch.tensor(0.0, device=pred.device)
    for pd, td in zip(_grid_gradients(pred[:, 0:1]), _grid_gradients(target[:, 0:1])):
        gp = gp + mse(pd, td)
    for sd, td in zip(_grid_gradients(pred[:, 1:2]), _grid_gradients(target[:, 1:2])):
        gs = gs + mse(sd, td)

    total = (
        float(weights.get("pressure", 1.0)) * lp
        + float(weights.get("water_sat", 1.0)) * ls
        + float(weights.get("pressured", 0.0)) * gp
        + float(weights.get("saturationd", 0.0)) * gs
    )
    return total, lp, ls, gp, gs


def run_training(
    max_steps: int,
    batch_size: int,
    spatial_dim: int,
    in_channels: int,
    shape,
    seed: int,
    lr: float,
    log_every: int,
    weights: dict[str, float],
    output_dir: str,
    save_every: int,
    device: str,
    pressure_scale: float,
    saturation_scale: float,
    use_unrst_baseline: bool,
    unrst_file: str,
) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)

    torch_device = torch.device(device)

    model = _MLP(in_channels=in_channels, width=48 if spatial_dim == 2 else 32, spatial_dim=spatial_dim).to(torch_device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    project_root = _find_project_root(Path(__file__).resolve().parent)
    baseline_x, baseline_y, baseline_used = (None, None, "")
    if use_unrst_baseline:
        bx, by, used = _load_unrst_baseline_batch(
            project_root=project_root,
            spatial_dim=spatial_dim,
            shape=shape,
            in_channels=in_channels,
            baseline_file=unrst_file,
        )
        baseline_x, baseline_y, baseline_used = bx, by, used

    if baseline_x is not None and baseline_y is not None:
        # Warm-start the final bias to baseline means so distribution starts closer to reservoir states.
        with torch.no_grad():
            final = model.net[-1]
            if hasattr(final, "weight") and hasattr(final, "bias"):
                final.weight.zero_()
                mean_p = float(baseline_y[:, 0:1].mean().item())
                mean_s = float(baseline_y[:, 1:2].mean().item())
                final.bias[0] = mean_p
                final.bias[1] = mean_s
        print(f"baseline_mode=enabled source={baseline_used}")
    else:
        print("baseline_mode=disabled source=synthetic")

    batch_gen = torch.Generator().manual_seed(seed + 2026)

    for step in range(max_steps):
        if baseline_x is not None and baseline_y is not None:
            n = baseline_x.shape[0]
            idx = torch.randint(0, n, (batch_size,), generator=batch_gen)
            x = baseline_x[idx]
            y = baseline_y[idx]
        else:
            x, y = _synthetic_grid_batch(
                batch_size,
                in_channels,
                shape,
                seed + step,
                pressure_scale=pressure_scale,
                saturation_scale=saturation_scale,
            )
        x = x.to(torch_device)
        y = y.to(torch_device)
        pred = model(x)
        loss, lp, ls, gp, gs = _loss_components(pred, y, weights)
        opt.zero_grad()
        loss.backward()
        opt.step()

        if save_every > 0 and (step % save_every == 0 or step == max_steps - 1):
            ckpt_path = out_dir / f"compat_checkpoint_step_{step:06d}.pt"
            torch.save(
                {
                    "step": step,
                    "model": model.state_dict(),
                    "optimizer": opt.state_dict(),
                    "shape": shape,
                    "in_channels": in_channels,
                    "spatial_dim": spatial_dim,
                },
                ckpt_path,
            )

            pred_path = out_dir / f"compat_prediction_step_{step:06d}.npz"
            np.savez_compressed(
                pred_path,
                pred=pred.detach().cpu().numpy(),
                target=y.detach().cpu().numpy(),
            )

        if step % log_every == 0 or step == max_steps - 1:
            print(
                "step={:06d} total={:.6e} p={:.6e} s={:.6e} pd={:.6e} sd={:.6e}".format(
                    step,
                    loss.item(),
                    lp.item(),
                    ls.item(),
                    gp.item(),
                    gs.item(),
                )
            )


def run_training_from_cfg(
    cfg,
    default_dim: int,
    workflow_name: str | None = None,
    scenario_name: str | None = None,
) -> None:
    seed = int(_cfg_get(cfg, 'seed', 1234))
    max_steps = int(_cfg_get(cfg, 'training.max_steps', 50))
    batch_size = int(_cfg_get(cfg, 'custom.NVRS.batch_size', 512))
    if batch_size <= 0:
        batch_size = 512
    nx = int(_cfg_get(cfg, 'custom.NVRS.nx', 33))
    ny = int(_cfg_get(cfg, 'custom.NVRS.ny', 33))
    nz = int(_cfg_get(cfg, 'custom.NVRS.nz', 1))
    in_channels = int(_cfg_get(cfg, 'custom.NVRS.input_channel', 7))
    profile = _workflow_profile(workflow_name, scenario_name)
    lr = float(_cfg_get(cfg, 'optimizer.lr', 1e-3)) * float(profile["lr_scale"])
    rec = int(_cfg_get(cfg, 'training.rec_constraint_freq', 0))
    log_every = max(1, rec if rec > 0 else max_steps // 5)
    network_dir = str(_cfg_get(cfg, 'network_dir', 'compat_outputs'))
    save_every = int(_cfg_get(cfg, 'training.rec_results_freq', 0))
    if save_every <= 0:
        save_every = max(1, max_steps)
    loss_weights = {
        'pressure': float(_cfg_get(cfg, 'loss.weights.pressure', 1.0)),
        'water_sat': float(_cfg_get(cfg, 'loss.weights.water_sat', 1.0)),
        'pressured': float(_cfg_get(cfg, 'loss.weights.pressured', 0.0)),
        'saturationd': float(_cfg_get(cfg, 'loss.weights.saturationd', 0.0)),
    }
    for key, scale in profile["loss_scale"].items():
        loss_weights[key] = float(loss_weights[key]) * float(scale)
    use_cuda = bool(_cfg_get(cfg, 'use_cuda', False))
    device = 'cuda' if use_cuda and torch.cuda.is_available() else 'cpu'
    pressure_scale = float(_cfg_get(cfg, 'custom.NVRS.pini_alt', 1000.0))
    if pressure_scale <= 0:
        pressure_scale = 1000.0
    saturation_scale = float(_cfg_get(cfg, 'custom.NVRS.IWSw', 0.2))
    if saturation_scale <= 0:
        saturation_scale = 0.2
    use_unrst_baseline = bool(_cfg_get(cfg, 'custom.baseline.use_unrst', True))
    project_root = _find_project_root(Path(__file__).resolve().parent)
    explicit_baseline = str(_cfg_get(cfg, 'custom.baseline.unrst_path', ''))
    unrst_file = _select_baseline_path(project_root, explicit_baseline, profile["baseline_candidates"])
    run_training(
        max_steps=max_steps,
        batch_size=batch_size,
        spatial_dim=default_dim,
        in_channels=in_channels,
        shape=_grid_shape(default_dim, nx, ny, nz),
        seed=seed,
        lr=lr,
        log_every=log_every,
        weights=loss_weights,
        output_dir=network_dir,
        save_every=save_every,
        device=device,
        pressure_scale=pressure_scale,
        saturation_scale=saturation_scale,
        use_unrst_baseline=use_unrst_baseline,
        unrst_file=unrst_file,
    )


def run_compare(default_dim: int) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--surrogate', type=int, default=1, choices=[1, 2, 3, 4])
    parser.add_argument('--samples', type=int, default=256)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--dim', type=int, default=default_dim, choices=[2, 3])
    parser.add_argument('--nx', type=int, default=33)
    parser.add_argument('--ny', type=int, default=33)
    parser.add_argument('--nz', type=int, default=1)
    parser.add_argument('--in-channels', type=int, default=7)
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--pressure-scale', type=float, default=1000.0)
    parser.add_argument('--saturation-scale', type=float, default=0.2)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    shape = _grid_shape(args.dim, args.nx, args.ny, args.nz)
    x, y_full = _synthetic_grid_batch(
        args.samples,
        args.in_channels,
        shape,
        args.seed,
        pressure_scale=args.pressure_scale,
        saturation_scale=args.saturation_scale,
    )
    torch_device = torch.device('cuda' if args.device == 'cuda' and torch.cuda.is_available() else 'cpu')
    x = x.to(torch_device)
    y_full = y_full.to(torch_device)
    y = y_full[:, 0:1]
    mse = nn.MSELoss()

    widths = [16, 24, 32, 48]
    models = []
    for w in widths:
        m = _MLP(in_channels=args.in_channels, width=w, spatial_dim=args.dim).to(torch_device)
        head = nn.Sequential(m.net[0], m.net[1], m.net[2], m.net[3], m.net[4])
        models.append(head)
    losses = []
    for i, model in enumerate(models, start=1):
        with torch.no_grad():
            pred = model(x)[:, 0:1]
            l = mse(pred, y).item()
        losses.append(l)
        print(f"surrogate_{i}_mse={l:.6e}")

    idx = args.surrogate - 1
    print(f"selected_surrogate={args.surrogate} selected_mse={losses[idx]:.6e}")
