# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pyvista as pv


def save_polyvtk(points, file_stem, **point_data):
    """Save an unstructured point cloud to <file_stem>.vtp."""
    cloud = pv.PolyData(points)
    for name, values in point_data.items():
        cloud[name] = np.asarray(values)
    cloud.save(f"{file_stem}.vtp")


def sample_surface_points_from_mesh(mesh, nr_points, seed=0):
    """Sample boundary points by resampling surface vertices."""
    rng = np.random.default_rng(seed)
    surface = mesh.extract_surface()
    pts = np.asarray(surface.points)
    if pts.shape[0] == 0:
        raise ValueError("Mesh has no surface points")
    idx = rng.integers(0, pts.shape[0], size=nr_points)
    sampled = pts[idx]
    approx_area = float(surface.area) / float(max(nr_points, 1))
    return sampled, np.full((nr_points,), approx_area, dtype=np.float32)


def sample_interior_points_from_mesh(mesh, nr_points, seed=0, max_trials=50):
    """Sample interior points using rejection in bounding box + enclosed test."""
    rng = np.random.default_rng(seed)
    bounds = mesh.bounds
    mins = np.array([bounds[0], bounds[2], bounds[4]], dtype=np.float32)
    maxs = np.array([bounds[1], bounds[3], bounds[5]], dtype=np.float32)
    surface = mesh.extract_surface()

    kept = []
    trials = 0
    while sum(arr.shape[0] for arr in kept) < nr_points and trials < max_trials:
        need = nr_points - sum(arr.shape[0] for arr in kept)
        candidate = rng.uniform(mins, maxs, size=(max(need * 3, 1024), 3)).astype(np.float32)
        candidate_cloud = pv.PolyData(candidate)
        selected = candidate_cloud.select_enclosed_points(
            surface, check_surface=False, tolerance=0.0
        )
        mask = np.asarray(selected["SelectedPoints"]).astype(bool)
        inside = candidate[mask]
        if inside.shape[0] > 0:
            kept.append(inside)
        trials += 1

    if not kept:
        raise RuntimeError("Could not sample interior points from mesh")

    pts = np.concatenate(kept, axis=0)[:nr_points]
    approx_volume = float(mesh.volume) / float(max(nr_points, 1))
    return pts, np.full((nr_points,), approx_volume, dtype=np.float32)


def points_in_polygon(points, polygon):
    """Vectorized ray casting for 2D point-in-polygon."""
    x = points[:, 0]
    y = points[:, 1]
    poly = np.asarray(polygon, dtype=np.float64)
    inside = np.zeros(points.shape[0], dtype=bool)

    j = poly.shape[0] - 1
    for i in range(poly.shape[0]):
        xi, yi = poly[i, 0], poly[i, 1]
        xj, yj = poly[j, 0], poly[j, 1]
        cond = (yi > y) != (yj > y)
        x_intersect = (xj - xi) * (y - yi) / (yj - yi + 1.0e-12) + xi
        inside ^= cond & (x < x_intersect)
        j = i
    return inside


def sample_polygon_boundary(polygon, nr_points, seed=0):
    rng = np.random.default_rng(seed)
    poly = np.asarray(polygon, dtype=np.float32)
    nxt = np.roll(poly, -1, axis=0)
    edges = nxt - poly
    lengths = np.linalg.norm(edges, axis=1)
    probs = lengths / np.sum(lengths)
    edge_idx = rng.choice(len(edges), size=nr_points, p=probs)
    alpha = rng.random((nr_points, 1), dtype=np.float32)
    pts2d = poly[edge_idx] + alpha * edges[edge_idx]
    pts = np.column_stack([pts2d, np.zeros((nr_points,), dtype=np.float32)])
    approx_len = float(np.sum(lengths)) / float(max(nr_points, 1))
    return pts, np.full((nr_points,), approx_len, dtype=np.float32)


def sample_polygon_interior(polygon, nr_points, seed=0, max_trials=50):
    rng = np.random.default_rng(seed)
    poly = np.asarray(polygon, dtype=np.float32)
    mins = np.min(poly, axis=0)
    maxs = np.max(poly, axis=0)

    kept = []
    trials = 0
    while sum(arr.shape[0] for arr in kept) < nr_points and trials < max_trials:
        need = nr_points - sum(arr.shape[0] for arr in kept)
        candidate = rng.uniform(mins, maxs, size=(max(need * 3, 1024), 2)).astype(np.float32)
        mask = points_in_polygon(candidate, poly)
        inside = candidate[mask]
        if inside.shape[0] > 0:
            kept.append(inside)
        trials += 1

    if not kept:
        raise RuntimeError("Could not sample interior points from polygon")

    pts2d = np.concatenate(kept, axis=0)[:nr_points]
    pts = np.column_stack([pts2d, np.zeros((nr_points,), dtype=np.float32)])
    area = _polygon_area(poly)
    approx_area = float(area) / float(max(nr_points, 1))
    return pts, np.full((nr_points,), approx_area, dtype=np.float32)


def _polygon_area(poly):
    x = poly[:, 0]
    y = poly[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
