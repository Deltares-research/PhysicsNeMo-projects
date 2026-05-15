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

import numpy as np
import pyvista as pv
import warnings

from v2_geometry_utils import (
    save_polyvtk,
    sample_interior_points_from_mesh,
    sample_surface_points_from_mesh,
)

if __name__ == "__main__":
    nr_points = 100000

    stl_path = "./stl_files/tessellated_example.stl"
    try:
        mesh = pv.read(stl_path).triangulate()
        if mesh.n_points == 0:
            raise ValueError("STL contains no geometry (possibly Git LFS pointer)")
    except Exception as exc:
        warnings.warn(
            f"Failed to load {stl_path} ({exc}). Using synthetic fallback mesh."
        )
        mesh = pv.Sphere(radius=1.0, theta_resolution=64, phi_resolution=64).triangulate()

    mesh = mesh.clip(normal=(1.0, 0.0, 0.0), origin=(0.0, 0.0, 0.0), invert=False)

    boundary_pts, boundary_area = sample_surface_points_from_mesh(
        mesh, nr_points=nr_points, seed=1
    )
    save_polyvtk(boundary_pts, "tessellated_boundary", area=boundary_area)
    print("Surface Area (mesh): {:.3f}".format(float(mesh.area)))

    interior_pts, interior_volume = sample_interior_points_from_mesh(
        mesh, nr_points=nr_points, seed=2
    )
    save_polyvtk(interior_pts, "tessellated_interior", area=interior_volume)
    print("Volume (mesh): {:.3f}".format(float(mesh.volume)))
