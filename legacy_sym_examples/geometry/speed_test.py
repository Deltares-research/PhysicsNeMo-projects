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
import time
import warnings

from v2_geometry_utils import (
    save_polyvtk,
    sample_interior_points_from_mesh,
    sample_surface_points_from_mesh,
)


def sdf_box(p, half_size):
    q = np.abs(p) - half_size
    outside = np.linalg.norm(np.maximum(q, 0.0), axis=1)
    inside = np.minimum(np.maximum.reduce(q.T), 0.0)
    return outside + inside


def sdf_sphere(p, radius):
    return np.linalg.norm(p, axis=1) - radius


def sdf_cylinder_axis(p, radius, half_height, axis="z"):
    if axis == "z":
        radial = np.linalg.norm(p[:, :2], axis=1)
        axial = np.abs(p[:, 2])
    elif axis == "x":
        radial = np.linalg.norm(p[:, 1:], axis=1)
        axial = np.abs(p[:, 0])
    else:
        radial = np.linalg.norm(p[:, [0, 2]], axis=1)
        axial = np.abs(p[:, 1])
    d = np.stack([radial - radius, axial - half_height], axis=1)
    return np.minimum(np.maximum(d[:, 0], d[:, 1]), 0.0) + np.linalg.norm(
        np.maximum(d, 0.0), axis=1
    )


def sample_implicit_csg(nr_points, boundary=False, seed=0):
    rng = np.random.default_rng(seed)
    pts = []
    while sum(x.shape[0] for x in pts) < nr_points:
        candidate = rng.uniform(-1.5, 1.5, size=(max(8192, nr_points), 3)).astype(
            np.float32
        )
        box_sdf = sdf_box(candidate, np.array([1.0, 1.0, 1.0], dtype=np.float32))
        sphere_sdf = sdf_sphere(candidate, 1.2)
        base = np.maximum(box_sdf, sphere_sdf)
        c1 = sdf_cylinder_axis(candidate, 0.5, 1.0, axis="z")
        c2 = sdf_cylinder_axis(candidate, 0.5, 1.0, axis="x")
        c3 = sdf_cylinder_axis(candidate, 0.5, 1.0, axis="y")
        sdf = np.maximum(base, -np.minimum(np.minimum(c1, c2), c3))
        if boundary:
            mask = np.abs(sdf) < 0.015
        else:
            mask = sdf <= 0.0
        chosen = candidate[mask]
        if chosen.shape[0] > 0:
            pts.append(chosen)
    return np.concatenate(pts, axis=0)[:nr_points]


def speed_check_mesh(mesh, nr_points):
    tic = time.time()
    s, area = sample_surface_points_from_mesh(mesh, nr_points=nr_points)
    surface_sample_time = time.time() - tic
    save_polyvtk(s, "boundary", area=area)
    tic = time.time()
    s, volume = sample_interior_points_from_mesh(mesh, nr_points=nr_points)
    volume_sample_time = time.time() - tic
    save_polyvtk(s, "interior", area=volume)
    print(
        "Surface sample (seconds per million point): {:.3e}".format(
            1000000 * surface_sample_time / nr_points
        )
    )
    print(
        "Volume sample (seconds per million point): {:.3e}".format(
            1000000 * volume_sample_time / nr_points
        )
    )


def speed_check_implicit(nr_points):
    tic = time.time()
    b = sample_implicit_csg(nr_points=nr_points, boundary=True)
    surface_sample_time = time.time() - tic
    save_polyvtk(b, "boundary")
    tic = time.time()
    i = sample_implicit_csg(nr_points=nr_points, boundary=False)
    volume_sample_time = time.time() - tic
    save_polyvtk(i, "interior")
    print(
        "Surface sample (seconds per million point): {:.3e}".format(
            1000000 * surface_sample_time / nr_points
        )
    )
    print(
        "Volume sample (seconds per million point): {:.3e}".format(
            1000000 * volume_sample_time / nr_points
        )
    )


if __name__ == "__main__":
    nr_points = 1000000

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

    print("Tesselated Speed Test")
    print("Number of triangles: {:d}".format(int(mesh.n_cells)))
    speed_check_mesh(mesh, nr_points)

    print("CSG Speed Test")
    speed_check_implicit(nr_points)

    # make boxes for many body check
    nr_boxes = [10, 100, 500]
    boxes = []
    for i in range(max(nr_boxes)):
        x_pos = (np.sqrt(5.0) * i % 0.8) + 0.1
        y_pos = (np.sqrt(3.0) * i % 0.8) + 0.1
        z_pos = (np.sqrt(7.0) * i % 0.8) + 0.1
        boxes.append(
            np.array(
                [[x_pos, x_pos + 0.05], [y_pos, y_pos + 0.05], [z_pos, z_pos + 0.05]]
            )
        )
    boxes = np.array(boxes)

    for nr_b in nr_boxes:
        print("CSG Many Box Speed Test, Number of Boxes " + str(nr_b))
        tic = time.time()
        rng = np.random.default_rng(1234)
        candidate = rng.uniform(0.0, 1.0, size=(nr_points * 2, 3)).astype(np.float32)
        inside = np.ones((candidate.shape[0],), dtype=bool)
        for i in range(nr_b):
            in_box = np.all(
                (candidate >= boxes[i, :, 0]) & (candidate <= boxes[i, :, 1]), axis=1
            )
            inside &= ~in_box
        cloud = candidate[inside][:nr_points]
        elapsed = time.time() - tic
        save_polyvtk(cloud, f"many_box_interior_{nr_b}")
        print(
            "Volume sample (seconds per million point): {:.3e}".format(
                1000000 * elapsed / nr_points
            )
        )
