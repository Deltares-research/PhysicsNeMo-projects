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

import glob
import numpy as np
import pyvista as pv
import warnings

from v2_geometry_utils import (
    save_polyvtk,
    sample_interior_points_from_mesh,
    sample_surface_points_from_mesh,
)

if __name__ == "__main__":
    bracket_files = glob.glob("./bracket_stl/*.stl")
    bracket_files.sort()

    meshes = []
    radius = []
    width = []
    for f in bracket_files:
        radius.append(float(f.split("_")[3]))
        width.append(float(f.split("_")[5][:-4]))
        try:
            m = pv.read(f).triangulate()
            if m.n_points == 0:
                raise ValueError("STL contains no geometry (possibly Git LFS pointer)")
        except Exception as exc:
            warnings.warn(f"Failed to load {f} ({exc}). Using synthetic fallback mesh.")
            m = pv.Cube(x_length=1.0, y_length=0.4, z_length=0.2).triangulate()
        meshes.append(m)

    if not meshes:
        raise RuntimeError("No STL files found in ./bracket_stl")

    nr_points = 1000000
    per_mesh = max(1000, nr_points // len(meshes))
    boundary_parts = []
    interior_parts = []
    radius_parts = []
    width_parts = []

    for i, mesh in enumerate(meshes):
        bpts, _ = sample_surface_points_from_mesh(mesh, per_mesh, seed=10 + i)
        ipts, _ = sample_interior_points_from_mesh(mesh, per_mesh, seed=100 + i)
        boundary_parts.append(bpts)
        interior_parts.append(ipts)
        radius_parts.append(np.full((bpts.shape[0],), radius[i], dtype=np.float32))
        width_parts.append(np.full((bpts.shape[0],), width[i], dtype=np.float32))

    boundary = np.concatenate(boundary_parts, axis=0)[:nr_points]
    interior = np.concatenate(interior_parts, axis=0)[:nr_points]

    radius_b = np.concatenate(radius_parts, axis=0)[: boundary.shape[0]]
    width_b = np.concatenate(width_parts, axis=0)[: boundary.shape[0]]
    save_polyvtk(boundary, "parameterized_bracket_boundary", radius=radius_b, width=width_b)

    # Reuse first parameter pair for interior metadata size consistency.
    save_polyvtk(
        interior,
        "parameterized_bracket_interior",
        radius=np.full((interior.shape[0],), radius[0], dtype=np.float32),
        width=np.full((interior.shape[0],), width[0], dtype=np.float32),
    )
