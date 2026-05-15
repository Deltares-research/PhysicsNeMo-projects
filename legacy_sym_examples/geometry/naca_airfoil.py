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

from v2_geometry_utils import (
    save_polyvtk,
    sample_polygon_boundary,
    sample_polygon_interior,
)


# Naca implementation modified from https://stackoverflow.com/questions/31815041/plotting-a-naca-4-series-airfoil
# https://en.wikipedia.org/wiki/NACA_airfoil#Equation_for_a_cambered_4-digit_NACA_airfoil
def camber_line(x, m, p, c):
    x = np.asarray(x, dtype=np.float64)
    cl = np.empty_like(x)
    mask = x <= (c * p)
    cl[mask] = m * (x[mask] / p**2) * (2.0 * p - (x[mask] / c))
    cl[~mask] = m * ((c - x[~mask]) / (1 - p) ** 2) * (1.0 + (x[~mask] / c) - 2.0 * p)
    return cl


def dyc_over_dx(x, m, p, c):
    x = np.asarray(x, dtype=np.float64)
    dy = np.empty_like(x)
    mask = x <= (c * p)
    dy[mask] = ((2.0 * m) / p**2) * (p - x[mask] / c)
    dy[~mask] = (2.0 * m) / ((1 - p) ** 2) * (p - x[~mask] / c)
    return np.arctan(dy)


def thickness(x, t, c):
    x = np.asarray(x, dtype=np.float64)
    term1 = 0.2969 * np.sqrt(x / c)
    term2 = -0.1260 * (x / c)
    term3 = -0.3516 * (x / c) ** 2
    term4 = 0.2843 * (x / c) ** 3
    term5 = -0.1015 * (x / c) ** 4
    return 5.0 * t * c * (term1 + term2 + term3 + term4 + term5)


def naca4(x, m, p, t, c=1):
    x = np.asarray(x, dtype=np.float64)
    th = dyc_over_dx(x, m, p, c)
    yt = thickness(x, t, c)
    yc = camber_line(x, m, p, c)

    upper = np.column_stack([x - yt * np.sin(th), yc + yt * np.cos(th)])
    lower = np.column_stack([x[::-1] + yt[::-1] * np.sin(th[::-1]), yc[::-1] - yt[::-1] * np.cos(th[::-1])])
    return np.vstack([upper, lower])


if __name__ == "__main__":
    m = 0.02
    p = 0.4
    t = 0.12
    c = 1.0

    x = [x for x in np.linspace(0, 0.2, 10)] + [x for x in np.linspace(0.2, 1.0, 10)][
        1:
    ]  # higher res in front
    polygon = naca4(x, m, p, t, c)[:-1].astype(np.float32)

    boundary, b_area = sample_polygon_boundary(polygon, nr_points=100000, seed=0)
    interior, i_area = sample_polygon_interior(polygon, nr_points=100000, seed=1)

    save_polyvtk(boundary, "naca_boundary", area=b_area)
    save_polyvtk(interior, "naca_interior", area=i_area)
