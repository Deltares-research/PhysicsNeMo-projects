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

from v2_geometry_utils import save_polyvtk, sample_polygon_boundary, sample_polygon_interior


def make_hole_polygon(y_pos, radius=0.3, n=256):
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return np.column_stack([radius * np.cos(theta), y_pos + radius * np.sin(theta)])


def sample_plate_minus_circle(nr_points, y_pos, seed=0):
    rng = np.random.default_rng(seed)
    hole = make_hole_polygon(y_pos=y_pos)

    # Boundary is union of outer rectangle + hole ring.
    n_rect = nr_points // 2
    rect_poly = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]], dtype=np.float32)
    b_rect, _ = sample_polygon_boundary(rect_poly, n_rect, seed=seed)
    b_hole, _ = sample_polygon_boundary(hole, nr_points - n_rect, seed=seed + 1)
    boundary = np.concatenate([b_rect, b_hole], axis=0)

    kept = []
    while sum(x.shape[0] for x in kept) < nr_points:
        candidate = rng.uniform(-1.0, 1.0, size=(max(4096, nr_points), 2)).astype(np.float32)
        dist_to_hole_center = np.sqrt(candidate[:, 0] ** 2 + (candidate[:, 1] - y_pos) ** 2)
        inside = dist_to_hole_center >= 0.3
        chosen = candidate[inside]
        if chosen.shape[0] > 0:
            kept.append(chosen)
    interior_2d = np.concatenate(kept, axis=0)[:nr_points]
    interior = np.column_stack([interior_2d, np.zeros((nr_points,), dtype=np.float32)])

    return boundary, interior

if __name__ == "__main__":
    nr_points = 100000

    # Approximate full parameter range by mixing several y positions.
    mixed_boundary = []
    mixed_interior = []
    for i, y_pos in enumerate(np.linspace(-0.8, 0.8, 5)):
        b, s = sample_plate_minus_circle(nr_points // 5, float(y_pos), seed=100 + i)
        mixed_boundary.append(b)
        mixed_interior.append(s)
    save_polyvtk(np.concatenate(mixed_boundary, axis=0), "parameterized_boundary")
    save_polyvtk(np.concatenate(mixed_interior, axis=0), "parameterized_interior")

    b0, s0 = sample_plate_minus_circle(nr_points, y_pos=0.0, seed=7)
    save_polyvtk(b0, "y_pos_zero_boundary")
    save_polyvtk(s0, "y_pos_zero_interior")
