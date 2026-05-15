# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import hydra
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sympy import Symbol, Function, Number

from hydra.utils import to_absolute_path
from omegaconf import DictConfig as PhysicsNeMoConfig
from physicsnemo.models.mlp.fully_connected import FullyConnected

try:
    from compat.pde_compat import PDE
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from compat.pde_compat import PDE


# Read in npz files generated using finite difference simulator Devito
def read_wf_data(time, dLen):
    file_path = "Training_data"
    if not os.path.exists(to_absolute_path(file_path)):
        warnings.warn(
            f"Directory {file_path} does not exist. Using synthetic fallback wavefield. For production runs, download supplemental files from NGC https://catalog.ngc.nvidia.com/orgs/nvidia/teams/physicsnemo/resources/physicsnemo_sym_examples_supplemental_materials"
        )
        n = 128
        yy, xx = np.meshgrid(
            np.linspace(0, dLen, n, dtype=np.float32),
            np.linspace(0, dLen, n, dtype=np.float32),
            indexing="ij",
        )
        tsec = time * 0.001
        sigma = 0.18 * dLen
        gauss = np.exp(-((xx - 0.5 * dLen) ** 2 + (yy - 0.5 * dLen) ** 2) / (2.0 * sigma**2))
        phase = np.cos(6.0 * np.pi * tsec)
        wave = (gauss * phase).astype(np.float32)
    else:
        wf_filename = to_absolute_path(f"Training_data/wf_{int(time):04d}ms.npz")
        wave = np.load(wf_filename)["arr_0"].astype(np.float32)
    mesh_y, mesh_x = np.meshgrid(
        np.linspace(0, dLen, wave.shape[0]),
        np.linspace(0, dLen, wave.shape[1]),
        indexing="ij",
    )
    invar = {}
    invar["x"] = np.expand_dims(mesh_y.astype(np.float32).flatten(), axis=-1)
    invar["y"] = np.expand_dims(mesh_x.astype(np.float32).flatten(), axis=-1)
    invar["t"] = np.full_like(invar["x"], time * 0.001)
    outvar = {}
    outvar["u"] = np.expand_dims(wave.flatten(), axis=-1)
    return invar, outvar


# define open boundary conditions
class OpenBoundary(PDE):
    """
    Open boundary condition for wave problems
    Ref: http://hplgit.github.io/wavebc/doc/pub/._wavebc_cyborg002.html

    Parameters
    ==========
    u : str
        The dependent variable.
    c : float, Sympy Symbol/Expr, str
        Wave speed coefficient. If `c` is a str then it is
        converted to Sympy Function of form 'c(x,y,z,t)'.
        If 'c' is a Sympy Symbol or Expression then this
        is substituted into the equation.
    dim : int
        Dimension of the wave equation (1, 2, or 3). Default is 2.
    time : bool
        If time-dependent equations or not. Default is True.
    """

    name = "OpenBoundary"

    def __init__(self, u="u", c="c", dim=3, time=True):
        # set params
        self.u = u
        self.dim = dim
        self.time = time

        # coordinates
        x, y, z = Symbol("x"), Symbol("y"), Symbol("z")

        # normal
        normal_x, normal_y, normal_z = (
            Symbol("normal_x"),
            Symbol("normal_y"),
            Symbol("normal_z"),
        )

        # time
        t = Symbol("t")

        # make input variables
        input_variables = {"x": x, "y": y, "z": z, "t": t}
        if self.dim == 1:
            input_variables.pop("y")
            input_variables.pop("z")
        elif self.dim == 2:
            input_variables.pop("z")
        if not self.time:
            input_variables.pop("t")

        # Scalar function
        assert type(u) == str, "u needs to be string"
        u = Function(u)(*input_variables)

        # wave speed coefficient
        if type(c) is str:
            c = Function(c)(*input_variables)
        elif type(c) in [float, int]:
            c = Number(c)

        # set equations
        self.equations = {}
        self.equations["open_boundary"] = (
            u.diff(t)
            + normal_x * c * u.diff(x)
            + normal_y * c * u.diff(y)
            + normal_z * c * u.diff(z)
        )


class WaveEquation2D(PDE):
    """2D acoustic wave PDE: u_tt - div(c^2 grad(u)) = 0."""

    name = "WaveEquation2D"

    def __init__(self, u="u", c="c"):
        self.dim = 2

        x, y, t = Symbol("x"), Symbol("y"), Symbol("t")
        iv = {"x": x, "y": y, "t": t}

        assert isinstance(u, str), "u needs to be string"
        u = Function(u)(*iv.values())

        if isinstance(c, str):
            c = Function(c)(*iv.values())
        elif isinstance(c, (float, int)):
            c = Number(c)

        self.equations = {
            "wave_equation": u.diff(t, 2) - (c**2 * u.diff(x)).diff(x) - (c**2 * u.diff(y)).diff(y)
        }


def wave_residual_2d(coords, u_pred, c_pred):
    du = torch.autograd.grad(
        u_pred,
        coords,
        grad_outputs=torch.ones_like(u_pred),
        create_graph=True,
    )[0]
    u_x = du[:, 0:1]
    u_y = du[:, 1:2]
    u_t = du[:, 2:3]

    d2u_dt = torch.autograd.grad(
        u_t,
        coords,
        grad_outputs=torch.ones_like(u_t),
        create_graph=True,
    )[0]
    u_tt = d2u_dt[:, 2:3]

    flux_x = (c_pred**2) * u_x
    flux_y = (c_pred**2) * u_y

    dflux_x = torch.autograd.grad(
        flux_x,
        coords,
        grad_outputs=torch.ones_like(flux_x),
        create_graph=True,
    )[0]
    dflux_y = torch.autograd.grad(
        flux_y,
        coords,
        grad_outputs=torch.ones_like(flux_y),
        create_graph=True,
    )[0]

    return u_tt - dflux_x[:, 0:1] - dflux_y[:, 1:2]


def build_wave_observation_pool(snapshot_times_ms, dlen):
    xs, ys, ts, us = [], [], [], []
    for ms in snapshot_times_ms:
        invar, outvar = read_wf_data(ms, dlen)
        xs.append(invar["x"])
        ys.append(invar["y"])
        ts.append(invar["t"])
        us.append(outvar["u"])
    return {
        "x": np.concatenate(xs, axis=0).astype(np.float32),
        "y": np.concatenate(ys, axis=0).astype(np.float32),
        "t": np.concatenate(ts, axis=0).astype(np.float32),
        "u": np.concatenate(us, axis=0).astype(np.float32),
    }


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def run(cfg: PhysicsNeMoConfig) -> None:
    torch.manual_seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dlen = float(cfg.domain.length_km)
    t_min = float(cfg.domain.t_min)
    t_max = float(cfg.domain.t_max)

    wave_net = FullyConnected(
        in_features=3,
        out_features=1,
        num_layers=int(cfg.model.num_layers),
        layer_size=int(cfg.model.layer_size),
    ).to(device)
    speed_net = FullyConnected(
        in_features=2,
        out_features=1,
        num_layers=int(cfg.model.num_layers),
        layer_size=int(cfg.model.layer_size),
    ).to(device)

    optimizer = torch.optim.Adam(
        list(wave_net.parameters()) + list(speed_net.parameters()),
        lr=float(cfg.optimizer.lr),
        betas=(float(cfg.optimizer.beta1), float(cfg.optimizer.beta2)),
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=float(cfg.scheduler.decay_rate)
    )

    train_times = [float(ms) for ms in cfg.data.train_snapshot_ms]
    val_times = [float(ms) for ms in cfg.data.val_snapshot_ms]
    train_pool = build_wave_observation_pool(train_times, dlen)
    val_pool = build_wave_observation_pool(val_times, dlen)

    speed_axis = np.linspace(0.0, dlen, int(cfg.data.speed_grid_points), dtype=np.float32)
    speed_mesh_x, speed_mesh_y = np.meshgrid(speed_axis, speed_axis, indexing="ij")
    speed_x = torch.from_numpy(speed_mesh_x.reshape(-1, 1)).to(device)
    speed_y = torch.from_numpy(speed_mesh_y.reshape(-1, 1)).to(device)
    speed_target = torch.from_numpy(
        (np.tanh(80.0 * (speed_y.cpu().numpy() - 1.0)) / 2.0 + 1.5).astype(np.float32)
    ).to(device)

    train_x = torch.from_numpy(train_pool["x"]).to(device)
    train_y = torch.from_numpy(train_pool["y"]).to(device)
    train_t = torch.from_numpy(train_pool["t"]).to(device)
    train_u = torch.from_numpy(train_pool["u"]).to(device)
    n_train = train_x.shape[0]

    val_x = torch.from_numpy(val_pool["x"]).to(device)
    val_y = torch.from_numpy(val_pool["y"]).to(device)
    val_t = torch.from_numpy(val_pool["t"]).to(device)
    val_u = torch.from_numpy(val_pool["u"]).to(device)

    for step in range(int(cfg.training.max_steps)):
        optimizer.zero_grad()

        idx_obs = torch.randint(0, n_train, (int(cfg.batch_size.observation),), device=device)
        x_obs = train_x[idx_obs]
        y_obs = train_y[idx_obs]
        t_obs = train_t[idx_obs]
        u_obs = train_u[idx_obs]

        coords_obs = torch.cat([x_obs, y_obs, t_obs], dim=1)
        u_pred_obs = wave_net(coords_obs)
        obs_loss = torch.mean((u_pred_obs - u_obs) ** 2)

        x_int = torch.rand(int(cfg.batch_size.interior), 1, device=device) * dlen
        y_int = torch.rand(int(cfg.batch_size.interior), 1, device=device) * dlen
        t_int = t_min + torch.rand(int(cfg.batch_size.interior), 1, device=device) * (
            t_max - t_min
        )
        coords_int = torch.cat([x_int, y_int, t_int], dim=1).requires_grad_(True)
        u_int = wave_net(coords_int)
        c_raw_int = speed_net(coords_int[:, :2])
        c_int = F.softplus(c_raw_int) + float(cfg.model.min_speed)
        wave_residual = wave_residual_2d(coords_int, u_int, c_int)
        pde_loss = torch.mean(wave_residual**2)

        idx_speed = torch.randint(
            0,
            speed_x.shape[0],
            (int(cfg.batch_size.speed_supervision),),
            device=device,
        )
        c_speed_pred = F.softplus(speed_net(torch.cat([speed_x[idx_speed], speed_y[idx_speed]], dim=1)))
        c_speed_pred = c_speed_pred + float(cfg.model.min_speed)
        speed_loss = torch.mean((c_speed_pred - speed_target[idx_speed]) ** 2)

        n_edge = int(cfg.batch_size.boundary)
        t_edge = t_min + torch.rand(n_edge, 1, device=device) * (t_max - t_min)
        x_edge = torch.rand(n_edge, 1, device=device) * dlen
        y_edge = torch.rand(n_edge, 1, device=device) * dlen
        edges = [
            torch.cat([torch.zeros_like(y_edge), y_edge, t_edge], dim=1),
            torch.cat([torch.full_like(y_edge, dlen), y_edge, t_edge], dim=1),
            torch.cat([x_edge, torch.zeros_like(x_edge), t_edge], dim=1),
            torch.cat([x_edge, torch.full_like(x_edge, dlen), t_edge], dim=1),
        ]
        bc_loss = 0.0
        for edge_coords in edges:
            bc_loss = bc_loss + torch.mean(wave_net(edge_coords) ** 2)
        bc_loss = bc_loss / 4.0

        loss = (
            float(cfg.loss_weights.observation) * obs_loss
            + float(cfg.loss_weights.pde) * pde_loss
            + float(cfg.loss_weights.speed) * speed_loss
            + float(cfg.loss_weights.boundary) * bc_loss
        )
        loss.backward()
        optimizer.step()
        scheduler.step()

        if step % int(cfg.training.log_every) == 0:
            with torch.no_grad():
                val_idx = torch.randint(
                    0,
                    val_x.shape[0],
                    (int(cfg.batch_size.validation),),
                    device=device,
                )
                val_coords = torch.cat([val_x[val_idx], val_y[val_idx], val_t[val_idx]], dim=1)
                val_pred = wave_net(val_coords)
                val_loss = torch.mean((val_pred - val_u[val_idx]) ** 2)
                mean_speed = (
                    F.softplus(speed_net(torch.cat([speed_x[:4096], speed_y[:4096]], dim=1)))
                    + float(cfg.model.min_speed)
                ).mean()
                print(
                    f"step={step:06d} loss={loss.item():.4e} "
                    f"obs={obs_loss.item():.4e} pde={pde_loss.item():.4e} "
                    f"speed={speed_loss.item():.4e} bc={bc_loss.item():.4e} "
                    f"val={val_loss.item():.4e} mean_c={mean_speed.item():.4f}"
                )


if __name__ == "__main__":
    run()
