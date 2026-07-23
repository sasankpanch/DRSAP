# Builds a watertight mesh directly from the RANSAC plane equations that
# Map_Cleaning.py exports to plane_manifest.json -- no re-running segmentation,
# no point clouds involved, just plane intersections. Assumes those plane fits
# are good; this is a ground-truth mesh generator for a well-behaved scan, not
# a robust-to-bad-data pipeline.

import os
import json
from dataclasses import dataclass
from collections import defaultdict

import numpy as np
import trimesh
import trimesh.repair

manifest_path = "segmented_output/plane_manifest.json"
output_dir = "segmented_output_mesh"

up_axis = np.array([0.0, 0.0, 1.0])
east_axis = np.array([1.0, 0.0, 0.0])
north_axis = np.array([0.0, 1.0, 0.0])


@dataclass(eq=False)                                 # eq=False: numpy fields make default __eq__ ambiguous
class Plane:
    id: int
    label: str
    normal: np.ndarray
    d: float
    centroid: np.ndarray
    inlier_count: int


def load_planes(path):
    with open(path) as f:
        data = json.load(f)
    return [Plane(id=e["id"], label=e["label"], normal=np.array(e["normal"], dtype=float),
                  d=float(e["d"]), centroid=np.array(e["centroid"], dtype=float),
                  inlier_count=e["inlier_count"]) for e in data]


def merge_planes(planes):
    total = sum(p.inlier_count for p in planes)
    normal = sum(p.normal * p.inlier_count for p in planes) / total
    normal /= np.linalg.norm(normal)
    centroid = sum(p.centroid * p.inlier_count for p in planes) / total
    d = float(-np.dot(normal, centroid))
    return Plane(id=-1, label=planes[0].label, normal=normal, d=d, centroid=centroid, inlier_count=total)


def dedupe_planes(planes, normal_tol_deg=5.0, offset_tol_m=0.15):
    n = len(planes)
    if n <= 1:
        return list(planes)

    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for i in range(n):
        for j in range(i + 1, n):
            cos_a = min(abs(np.dot(planes[i].normal, planes[j].normal)), 1.0)
            angle = np.degrees(np.arccos(cos_a))
            if angle < normal_tol_deg:
                offset_diff = abs(np.dot(planes[i].centroid, planes[i].normal)
                                   - np.dot(planes[j].centroid, planes[i].normal))
                if offset_diff < offset_tol_m:
                    union(i, j)

    groups = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)

    merged = []
    for idxs in groups.values():
        group = [planes[i] for i in idxs]
        merged.append(group[0] if len(group) == 1 else merge_planes(group))
    return merged


def intersect_three_planes(p1, p2, p3, det_tol=1e-3):
    A = np.array([p1.normal, p2.normal, p3.normal])
    b = np.array([-p1.d, -p2.d, -p3.d])
    if abs(np.linalg.det(A)) < det_tol:
        return None
    return np.linalg.solve(A, b)


def ring_is_monotonic(angles):
    """True iff the angles (already in candidate ring order) sweep the full
    circle exactly once with no reversal -- i.e. the ring is star-convex
    around its centroid in this order. Unwrap first: raw atan2 output wraps
    at +-180deg, which looks like a reversal even when the sweep is a
    perfectly valid monotonic ring that happens to cross that boundary."""
    unwrapped = np.unwrap(angles)
    closed = np.concatenate([unwrapped, unwrapped[:1] + 2 * np.pi])
    return bool(np.all(np.diff(closed) > 1e-9))


def polygon_is_simple(points_xy):
    """Brute-force O(n^2) non-adjacent segment intersection check -- cheap at
    room-corner vertex counts, and a defense-in-depth check beyond angle
    monotonicity (which can be fooled by near-duplicate/degenerate points)."""
    n = len(points_xy)

    def orientation(a, b, c):
        val = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        if abs(val) < 1e-9:
            return 0
        return 1 if val > 0 else 2

    def on_segment(a, b, c):
        return (min(a[0], b[0]) - 1e-9 <= c[0] <= max(a[0], b[0]) + 1e-9
                and min(a[1], b[1]) - 1e-9 <= c[1] <= max(a[1], b[1]) + 1e-9)

    def segments_intersect(p1, p2, p3, p4):
        o1, o2 = orientation(p1, p2, p3), orientation(p1, p2, p4)
        o3, o4 = orientation(p3, p4, p1), orientation(p3, p4, p2)
        if o1 != o2 and o3 != o4:
            return True
        if o1 == 0 and on_segment(p1, p2, p3):
            return True
        if o2 == 0 and on_segment(p1, p2, p4):
            return True
        if o3 == 0 and on_segment(p3, p4, p1):
            return True
        if o4 == 0 and on_segment(p3, p4, p2):
            return True
        return False

    for i in range(n):
        a1, a2 = points_xy[i], points_xy[(i + 1) % n]
        for j in range(i + 1, n):
            if j in (i, (i + 1) % n) or (j + 1) % n == i:
                continue
            b1, b2 = points_xy[j], points_xy[(j + 1) % n]
            if segments_intersect(a1, a2, b1, b2):
                return False
    return True


def build_polyhedron(floor_planes, ceiling_planes, wall_planes, up_axis, east_axis, north_axis,
                      parallel_tol_deg=10.0, bbox_slack=1.0):
    if not floor_planes or not ceiling_planes or len(wall_planes) < 3:
        return None, "insufficient planes (need floor, ceiling, and >=3 walls)"

    floor = floor_planes[0] if len(floor_planes) == 1 else merge_planes(floor_planes)
    ceiling = ceiling_planes[0] if len(ceiling_planes) == 1 else merge_planes(ceiling_planes)
    walls = wall_planes
    n = len(walls)

    center = np.mean([w.centroid for w in walls], axis=0)

    def angle_of(p):
        rel = p.centroid - center
        return np.arctan2(np.dot(rel, north_axis), np.dot(rel, east_axis))

    walls_sorted = sorted(walls, key=angle_of)

    # No raw points available here -- bound plausibility using plane centroids
    # instead, with a generous slack since centroids are far sparser than a
    # full point cloud's extent.
    all_centroids = np.vstack([w.centroid for w in walls_sorted] + [floor.centroid, ceiling.centroid])
    bbox_lo = all_centroids.min(axis=0) - bbox_slack
    bbox_hi = all_centroids.max(axis=0) + bbox_slack

    sin_tol = np.sin(np.radians(parallel_tol_deg))
    floor_vertices, ceiling_vertices = [], []
    for i in range(n):
        w1, w2 = walls_sorted[i], walls_sorted[(i + 1) % n]
        if np.linalg.norm(np.cross(w1.normal, w2.normal)) < sin_tol:
            return None, f"near-parallel adjacent walls at edge {i}"

        vf = intersect_three_planes(w1, w2, floor)
        vc = intersect_three_planes(w1, w2, ceiling)
        if vf is None or vc is None:
            return None, f"degenerate plane-triple intersection at edge {i}"
        if np.any(vf < bbox_lo) or np.any(vf > bbox_hi) or np.any(vc < bbox_lo) or np.any(vc > bbox_hi):
            return None, f"intersection vertex implausible (outside scan bbox) at edge {i}"

        floor_vertices.append(vf)
        ceiling_vertices.append(vc)

    floor_vertices = np.array(floor_vertices)
    ceiling_vertices = np.array(ceiling_vertices)

    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(floor_vertices[i] - floor_vertices[j]) < 0.05:
                return None, "near-duplicate floor vertices -- degenerate wall adjacency"

    centroid_xy = floor_vertices.mean(axis=0)
    floor_xy = np.array([[np.dot(v - centroid_xy, east_axis), np.dot(v - centroid_xy, north_axis)]
                          for v in floor_vertices])

    if np.any(np.linalg.norm(floor_xy, axis=1) < 1e-4):
        return None, "a floor vertex coincides with the room centroid"

    angles = np.arctan2(floor_xy[:, 1], floor_xy[:, 0])
    if not ring_is_monotonic(angles):
        return None, "footprint is not star-convex in wall-azimuth order"
    if not polygon_is_simple(floor_xy):
        return None, "footprint polygon self-intersects"

    centroid_floor_3d = floor_vertices.mean(axis=0)
    centroid_ceiling_3d = ceiling_vertices.mean(axis=0)

    vertices = list(floor_vertices) + list(ceiling_vertices) + [centroid_floor_3d, centroid_ceiling_3d]
    idx_floor = lambda i: i
    idx_ceiling = lambda i: n + i
    idx_c_floor, idx_c_ceiling = 2 * n, 2 * n + 1

    # Emit triangles in any consistent order per quad -- don't try to guess
    # per-face outward direction with a heuristic dot product, since two
    # triangles of a near-planar (but not exactly planar) real-world quad can
    # legitimately disagree on that by floating-point noise. Global winding
    # consistency is fixed below via trimesh's own edge-adjacency repair, and
    # the overall inside-out/outside-in sign is checked once at the end.
    faces = []
    for i in range(n):
        i1 = (i + 1) % n
        faces.append((idx_c_ceiling, idx_ceiling(i), idx_ceiling(i1)))
        faces.append((idx_c_floor, idx_floor(i1), idx_floor(i)))
        faces.append((idx_floor(i), idx_floor(i1), idx_ceiling(i1)))
        faces.append((idx_floor(i), idx_ceiling(i1), idx_ceiling(i)))

    mesh_out = trimesh.Trimesh(vertices=np.array(vertices), faces=np.array(faces), process=True)
    trimesh.repair.fix_winding(mesh_out)

    if not mesh_out.is_watertight:
        return None, "assembled mesh is not watertight"
    if not mesh_out.is_winding_consistent:
        return None, "assembled mesh winding could not be made consistent"

    if mesh_out.volume < 0:                    # consistent but inside-out -- flip once, globally
        mesh_out.invert()
    if mesh_out.volume <= 0:
        return None, f"non-positive mesh volume ({mesh_out.volume:.4f}) after winding repair"

    return mesh_out, None


print(f"Loading plane manifest from {manifest_path}...")
planes = load_planes(manifest_path)
print(f" -> {len(planes)} planes loaded")

floor_planes = [p for p in planes if p.label == "floor"]
ceiling_planes = [p for p in planes if p.label == "ceiling"]
wall_planes = [p for p in planes if p.label == "wall"]

print(f"\nDeduplicating {len(wall_planes)} wall plane(s)...")
wall_planes = dedupe_planes(wall_planes, normal_tol_deg=5.0, offset_tol_m=0.15)
print(f" -> {len(wall_planes)} distinct wall(s) after dedup")

print("\nBuilding mesh from plane equations...")
poly_mesh, fail_reason = build_polyhedron(floor_planes, ceiling_planes, wall_planes,
                                           up_axis, east_axis, north_axis)

if poly_mesh is not None:
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "Room_Plane_Mesh.ply")
    poly_mesh.export(out_path)
    print(f" -> Watertight: {poly_mesh.is_watertight}, winding consistent: {poly_mesh.is_winding_consistent}")
    print(f" -> Volume: {poly_mesh.volume:.3f} m^3")
    print(f" -> Saved mesh to {out_path}")
else:
    print(f" -> Could not build a mesh from these planes: {fail_reason}")

print("\nProcessing complete!")
