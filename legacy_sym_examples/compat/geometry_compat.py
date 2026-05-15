# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np


@dataclass(frozen=True)
class Parameter:
    name: str

    def __str__(self) -> str:
        return self.name


class Parameterization:
    def __init__(self, params: Optional[Dict[Any, Any]] = None):
        self.params = params or {}


class GeometryExpr:
    """Minimal CSG-style compatibility object used by legacy geometry scripts."""

    def __init__(self, op: str, lhs: Any = None, rhs: Any = None, meta: Optional[Dict[str, Any]] = None):
        self.op = op
        self.lhs = lhs
        self.rhs = rhs
        self.meta = meta or {}

    def __add__(self, other: Any):
        return GeometryExpr("union", self, other)

    def __sub__(self, other: Any):
        return GeometryExpr("difference", self, other)

    def __and__(self, other: Any):
        return GeometryExpr("intersection", self, other)

    def translate(self, shift):
        lower, upper = self.bounds()
        sx, sy, sz = shift
        return Box(
            (lower[0] + sx, lower[1] + sy, lower[2] + sz),
            (upper[0] + sx, upper[1] + sy, upper[2] + sz),
        )

    def rotate(self, angle, axis="z", center=None):
        _ = angle, axis, center
        return self

    def repeat(self, gap: Any, repeat_lower=None, repeat_higher=None, center=None):
        return GeometryExpr(
            "repeat",
            self,
            None,
            {
                "gap": gap,
                "repeat_lower": repeat_lower,
                "repeat_higher": repeat_higher,
                "center": center,
            },
        )

    @staticmethod
    def _num(v):
        if isinstance(v, (int, float, np.floating)):
            return float(v)
        try:
            return float(v)
        except Exception:
            return 0.0

    def bounds(self):
        if self.op in {"box", "channel", "plane"}:
            lower = tuple(self._num(v) for v in self.meta.get("lower", (0.0, 0.0, 0.0)))
            upper = tuple(self._num(v) for v in self.meta.get("upper", (1.0, 1.0, 1.0)))
            return lower, upper

        if self.op == "union":
            l1, u1 = self.lhs.bounds()
            l2, u2 = self.rhs.bounds()
            lower = (min(l1[0], l2[0]), min(l1[1], l2[1]), min(l1[2], l2[2]))
            upper = (max(u1[0], u2[0]), max(u1[1], u2[1]), max(u1[2], u2[2]))
            return lower, upper

        if self.op == "intersection":
            l1, u1 = self.lhs.bounds()
            l2, u2 = self.rhs.bounds()
            lower = (max(l1[0], l2[0]), max(l1[1], l2[1]), max(l1[2], l2[2]))
            upper = (min(u1[0], u2[0]), min(u1[1], u2[1]), min(u1[2], u2[2]))
            return lower, upper

        if self.op == "difference":
            return self.lhs.bounds()

        if self.op == "repeat":
            l, u = self.lhs.bounds()
            gap = self._num(self.meta.get("gap", 0.0))
            rh = self.meta.get("repeat_higher") or (0, 0, 0)
            k = self._num(rh[2])
            lower = l
            upper = (u[0], u[1], u[2] + gap * k)
            return lower, upper

        return (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)

    def sample_interior(self, n, bounds=None, parameterization=None):
        _ = parameterization
        lower, upper = self.bounds()
        if bounds:
            def get_pair(key, default_pair):
                if key in bounds:
                    return bounds[key]
                for k, v in bounds.items():
                    if str(k) == key:
                        return v
                return default_pair

            bx = get_pair("x", (lower[0], upper[0]))
            by = get_pair("y", (lower[1], upper[1]))
            bz = get_pair("z", (lower[2], upper[2]))
        else:
            bx, by, bz = (lower[0], upper[0]), (lower[1], upper[1]), (lower[2], upper[2])

        x = np.random.uniform(float(bx[0]), float(bx[1]), size=(n, 1)).astype(np.float32)
        y = np.random.uniform(float(by[0]), float(by[1]), size=(n, 1)).astype(np.float32)
        z = np.random.uniform(float(bz[0]), float(bz[1]), size=(n, 1)).astype(np.float32)
        vol = max((float(bx[1]) - float(bx[0])) * (float(by[1]) - float(by[0])) * (float(bz[1]) - float(bz[0])), 1.0e-8)
        area = np.full((n, 1), vol / max(n, 1), dtype=np.float32)
        return {"x": x, "y": y, "z": z, "area": area}

    def sample_boundary(self, n, parameterization=None):
        _ = parameterization
        lower, upper = self.bounds()
        if self.op == "plane":
            x0 = self._num(self.meta.get("lower", (0.0, 0.0, 0.0))[0])
            x = np.full((n, 1), x0, dtype=np.float32)
            y = np.random.uniform(lower[1], upper[1], size=(n, 1)).astype(np.float32)
            z = np.random.uniform(lower[2], upper[2], size=(n, 1)).astype(np.float32)
            area = np.full((n, 1), max((upper[1] - lower[1]) * (upper[2] - lower[2]), 1.0e-8) / max(n, 1), dtype=np.float32)
            return {"x": x, "y": y, "z": z, "area": area, "normal_x": np.ones((n, 1), dtype=np.float32)}

        return self.sample_interior(n)

    def sdf(self, invar, params):
        _ = params
        x = np.asarray(invar.get("x"), dtype=np.float32)
        y = np.asarray(invar.get("y"), dtype=np.float32)
        z = np.asarray(invar.get("z"), dtype=np.float32)
        lower, upper = self.bounds()

        dx = np.minimum(x - lower[0], upper[0] - x)
        dy = np.minimum(y - lower[1], upper[1] - y)
        dz = np.minimum(z - lower[2], upper[2] - z)
        inside = (x >= lower[0]) & (x <= upper[0]) & (y >= lower[1]) & (y <= upper[1]) & (z >= lower[2]) & (z <= upper[2])
        sdf_in = np.minimum(np.minimum(dx, dy), dz)
        sdf_out = -np.sqrt(
            np.maximum(lower[0] - x, 0.0) ** 2
            + np.maximum(x - upper[0], 0.0) ** 2
            + np.maximum(lower[1] - y, 0.0) ** 2
            + np.maximum(y - upper[1], 0.0) ** 2
            + np.maximum(lower[2] - z, 0.0) ** 2
            + np.maximum(z - upper[2], 0.0) ** 2
        )
        sdf = np.where(inside, sdf_in, sdf_out).astype(np.float32)
        return {"sdf": sdf}


class Box(GeometryExpr):
    def __init__(self, lower, upper, parameterization: Optional[Parameterization] = None):
        super().__init__("box", meta={"lower": lower, "upper": upper, "parameterization": parameterization})


class Channel(GeometryExpr):
    def __init__(self, lower, upper, parameterization: Optional[Parameterization] = None):
        super().__init__("channel", meta={"lower": lower, "upper": upper, "parameterization": parameterization})


class Channel2D(GeometryExpr):
    def __init__(self, lower, upper, parameterization: Optional[Parameterization] = None):
        l3 = (lower[0], lower[1], 0.0)
        u3 = (upper[0], upper[1], 0.0)
        super().__init__("channel2d", meta={"lower": l3, "upper": u3, "parameterization": parameterization})


class Plane(GeometryExpr):
    def __init__(self, lower, upper, normal, parameterization: Optional[Parameterization] = None):
        super().__init__(
            "plane",
            meta={"lower": lower, "upper": upper, "normal": normal, "parameterization": parameterization},
        )
