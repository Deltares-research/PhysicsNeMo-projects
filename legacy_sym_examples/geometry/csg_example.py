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

from v2_geometry_utils import save_polyvtk


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
    else:  # axis == "y"
        radial = np.linalg.norm(p[:, [0, 2]], axis=1)
        axial = np.abs(p[:, 1])

    d = np.stack([radial - radius, axial - half_height], axis=1)
    return np.minimum(np.maximum(d[:, 0], d[:, 1]), 0.0) + np.linalg.norm(
        np.maximum(d, 0.0), axis=1
    )


def sdf_csg(p):
    box_sdf = sdf_box(p, np.array([1.0, 1.0, 1.0], dtype=np.float32))
    sphere_sdf = sdf_sphere(p, radius=1.2)
    base = np.maximum(box_sdf, sphere_sdf)  # box & sphere

    c1 = sdf_cylinder_axis(p, radius=0.5, half_height=1.0, axis="z")
    c2 = sdf_cylinder_axis(p, radius=0.5, half_height=1.0, axis="x")
    c3 = sdf_cylinder_axis(p, radius=0.5, half_height=1.0, axis="y")
    cylinders = np.minimum(np.minimum(c1, c2), c3)

    return np.maximum(base, -cylinders)  # (box & sphere) - cylinders


def sample_shape(nr_points, boundary=False, seed=0):
    rng = np.random.default_rng(seed)
    pts = []
    while sum(x.shape[0] for x in pts) < nr_points:
        candidate = rng.uniform(-1.5, 1.5, size=(max(8192, nr_points), 3)).astype(
            np.float32
        )
        sdf = sdf_csg(candidate)
        if boundary:
            mask = np.abs(sdf) < 0.015
        else:
            mask = sdf <= 0.0
        chosen = candidate[mask]
        if chosen.shape[0] > 0:
            pts.append(chosen)
    return np.concatenate(pts, axis=0)[:nr_points]


def rotate_points(points, angle, axis):
    c = np.cos(angle)
    s = np.sin(angle)
    if axis == "z":
        r = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    elif axis == "y":
        r = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float32)
    else:
        r = np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float32)
    return points @ r.T

if __name__ == "__main__":
    nr_points = 100000

    boundary = sample_shape(nr_points=nr_points, boundary=True, seed=1)
    interior = sample_shape(nr_points=nr_points, boundary=False, seed=2)
    save_polyvtk(boundary, "boundary")
    save_polyvtk(interior, "interior")
    print("Boundary points: {:d}".format(boundary.shape[0]))
    print("Interior points: {:d}".format(interior.shape[0]))

    transformed_boundary = rotate_points(rotate_points(0.5 * boundary, np.pi / 4, "z"), np.pi / 4, "y")
    transformed_interior = rotate_points(rotate_points(0.5 * interior, np.pi / 4, "z"), np.pi / 4, "y")

    offsets = []
    for ix in (-1, 0, 1):
        for iy in (-1, 0, 1):
            for iz in (-1, 0, 1):
                offsets.append(np.array([4.0 * ix, 4.0 * iy, 4.0 * iz], dtype=np.float32))

    repeated_boundary = np.concatenate(
        [transformed_boundary + offset for offset in offsets], axis=0
    )
    repeated_interior = np.concatenate(
        [transformed_interior + offset for offset in offsets], axis=0
    )

    save_polyvtk(repeated_boundary, "repeated_boundary")
    save_polyvtk(repeated_interior, "repeated_interior")
    print("Repeated boundary points: {:d}".format(repeated_boundary.shape[0]))
    print("Repeated interior points: {:d}".format(repeated_interior.shape[0]))
