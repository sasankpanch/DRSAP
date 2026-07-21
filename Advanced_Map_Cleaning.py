# Advanced pipeline: Manhattan-frame rectification + analytic polyhedron volume.

import os
import json
from dataclasses import dataclass
from collections import defaultdict

import numpy as np
import open3d
import matplotlib.pyplot as plt
import trimesh
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

our_mesh = "PLTL-Room-Scan.ply"
baseline_mesh = "PLTL-Room-LIDAR-Scan.ply"
conference_mesh = "Conference_Room.ply"
no_cheese_mesh = "No_Cheese.ply"         
output_dir = "segmented_output_advanced"


@dataclass(eq=False)                                 # eq=False: numpy fields make default __eq__ ambiguous
class PlaneRecord:
    id: int
    source: str                                      # "primary" | "secondary" | "merged"
    normal: np.ndarray
    d: float
    centroid: np.ndarray
    inlier_count: int
    cloud: object                                     # open3d.geometry.PointCloud
    label: str = "unclassified"
    snapped: bool = False


# ---------------------------------------------------------------------------
# 1. Load, clean, RANSAC segment -- identical stages to Map_Cleaning.py
# ---------------------------------------------------------------------------

print("Loading mesh and Point Cloud")
mesh = open3d.io.read_triangle_mesh(our_mesh)
pcd = mesh.sample_points_uniformly(number_of_points=1000000)

print("Statiscal Outlier Removal..")
cl, stat_ind = pcd.remove_statistical_outlier(nb_neighbors=200, std_ratio=2.0)
pcd_clean = pcd.select_by_index(stat_ind)

print("Radius Outlier Removal..")
cl, rad_ind = pcd_clean.remove_radius_outlier(nb_points=50, radius=0.1)
pcd_clean = pcd_clean.select_by_index(rad_ind)

print("Voxel downsampling..")
voxel_downsized = pcd_clean.voxel_down_sample(voxel_size=0.03)

print("DBSCAN CLustering..")
labels = np.array(voxel_downsized.cluster_dbscan(eps=0.5, min_points=15, print_progress=True))
valid_indices = np.where(labels >= 0)[0]

if len(valid_indices) > 0:
    print(f"Keeping {len(np.unique(labels[labels >= 0]))} valid clusters.")
    pcd_final_clean = voxel_downsized.select_by_index(valid_indices)
else:
    print("DBSCAN couldn't find distinct clusters, saving downsized cloud instead...")
    pcd_final_clean = voxel_downsized

pcd_final_clean.estimate_normals(search_param=open3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
pcd_final_clean.orient_normals_consistent_tangent_plane(k=15)


def plane_segmentor(pcd, max_planes=6, dist_thres=0.05, ransac_n=3, num_iter=10000,
                     min_plane_ratio=0.005, normal_angle_thres=25):
    min_plane_points = max(100, int(min_plane_ratio * len(pcd.points)))
    remaining = pcd
    plane_models, plane_clouds = [], []

    for i in range(max_planes):
        if len(remaining.points) < min_plane_points:
            break

        plane_model, inliers = remaining.segment_plane(distance_threshold=dist_thres, ransac_n=ransac_n,
                                                         num_iterations=num_iter)
        if len(inliers) < min_plane_points:
            break

        a, b, c, d = plane_model
        plane_normal = np.array([a, b, c])
        plane_normal /= np.linalg.norm(plane_normal)

        inlier_normals = np.asarray(remaining.normals)[inliers]
        cos_angle = np.abs(inlier_normals @ plane_normal)
        angle_deg = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
        good_inliers = np.asarray(inliers)[angle_deg < normal_angle_thres]

        if len(good_inliers) < min_plane_points:
            break

        plane_cloud = remaining.select_by_index(good_inliers)
        plane_cloud.paint_uniform_color(plt.get_cmap("tab10")(i % 10)[:3])

        print(f"   Plane {i}: {len(good_inliers)} pts, "
              f"normal=({plane_normal[0]:.2f}, {plane_normal[1]:.2f}, {plane_normal[2]:.2f})")

        plane_models.append(plane_model)
        plane_clouds.append(plane_cloud)
        remaining = remaining.select_by_index(good_inliers, invert=True)

    return plane_models, plane_clouds, remaining


def build_records(plane_models, plane_clouds, source, start_id=0):
    records = []
    for i, (model, cloud) in enumerate(zip(plane_models, plane_clouds)):
        a, b, c, d = model
        normal = np.array([a, b, c])
        normal /= np.linalg.norm(normal)
        points = np.asarray(cloud.points)
        centroid = points.mean(axis=0)
        records.append(PlaneRecord(
            id=start_id + i, source=source, normal=normal, d=float(d),
            centroid=centroid, inlier_count=len(points), cloud=cloud,
        ))
    return records


print("RANSAC Plane Segmentation")
plane_models, plane_clouds, pcd_non_planar = plane_segmentor(
    pcd_final_clean, max_planes=12, dist_thres=0.05, ransac_n=3, num_iter=10000,
    min_plane_ratio=0.002, normal_angle_thres=25,
)

if not plane_clouds:
    raise SystemExit("no planes found -- check that segmentation kept enough structural points")

primary_records = build_records(plane_models, plane_clouds, source="primary")


# ---------------------------------------------------------------------------
# 2. Manhattan-frame rectification
# ---------------------------------------------------------------------------

def estimate_room_frame(records):
    """Geometric seed frame (up, east, north), no least-squares -- used to bootstrap
    the rotation fit and as the fallback frame if the fit can't run."""
    horiz = [r for r in records if abs(r.normal[2]) > 0.7]
    if not horiz:
        return None
    seed = max(horiz, key=lambda r: r.inlier_count)
    up = seed.normal if seed.normal[2] >= 0 else -seed.normal
    up = up / np.linalg.norm(up)

    vert = [r for r in records if abs(np.dot(r.normal, up)) < 0.4]
    if not vert:
        return None
    ref = max(vert, key=lambda r: r.inlier_count)
    east = ref.normal - np.dot(ref.normal, up) * up
    if np.linalg.norm(east) < 1e-6:
        return None
    east /= np.linalg.norm(east)
    north = np.cross(up, east)
    north /= np.linalg.norm(north)
    up = np.cross(east, north)                        # re-orthogonalize
    return up, east, north


def fit_manhattan_rotation(records, tol_deg=7.0):
    """Least-squares rotation aligning near-axis plane normals to canonical XYZ,
    weighted by inlier count. Planes already close to an axis (within tol_deg)
    supply the fit; nothing is force-fit onto a plane that isn't already close --
    that's what protects angled walls/bay windows from getting snapped to 90deg."""
    frame = estimate_room_frame(records)
    if frame is None:
        return None, None, None, None
    up0, east0, north0 = frame
    R0 = np.column_stack([east0, north0, up0]).T       # canonical ~= R0 @ original

    canon_axes = np.eye(3)
    cos_tol = np.cos(np.radians(tol_deg))
    targets, observed, weights = [], [], []
    for r in records:
        v = R0 @ r.normal
        idx = np.argmax(np.abs(v))
        cos_angle = abs(v[idx])
        if cos_angle >= cos_tol:
            targets.append(canon_axes[idx] * np.sign(v[idx]))
            observed.append(r.normal)
            weights.append(r.inlier_count)

    if len(targets) < 3:                                # not enough evidence for a robust fit
        return None, None, None, None

    rot, _ = Rotation.align_vectors(np.array(targets), np.array(observed),
                                     weights=np.array(weights, dtype=float))
    R = rot.as_matrix()
    up = R.T @ np.array([0.0, 0.0, 1.0])
    east = R.T @ np.array([1.0, 0.0, 0.0])
    north = R.T @ np.array([0.0, 1.0, 0.0])
    return R, up, east, north


def apply_manhattan_snap(records, R, tol_deg=7.0):
    """Mutates records: planes within tol_deg of a canonical axis (post-fit) get
    their normal/d replaced by an exact axis-aligned plane through the same
    centroid; others are left untouched and flagged unsnapped."""
    canon_axes = np.eye(3)
    cos_tol = np.cos(np.radians(tol_deg))
    for r in records:
        v = R @ r.normal
        idx = np.argmax(np.abs(v))
        cos_angle = abs(v[idx])
        if cos_angle >= cos_tol:
            snapped_canonical = canon_axes[idx] * np.sign(v[idx])
            snapped_original = R.T @ snapped_canonical
            r.normal = snapped_original
            r.d = float(-np.dot(snapped_original, r.centroid))
            r.snapped = True
        else:
            r.snapped = False


print("\nFitting Manhattan frame from primary plane normals...")
R_manhattan, up_axis, east_axis, north_axis = fit_manhattan_rotation(primary_records, tol_deg=7.0)

if R_manhattan is not None:
    apply_manhattan_snap(primary_records, R_manhattan, tol_deg=7.0)
    n_snapped = sum(1 for r in primary_records if r.snapped)
    print(f" -> Snapped {n_snapped}/{len(primary_records)} primary planes within 7 degrees of a canonical axis")
else:
    print(" -> Insufficient near-axis evidence for a Manhattan fit; using raw seed frame, no snapping applied")
    frame = estimate_room_frame(primary_records)
    if frame is not None:
        up_axis, east_axis, north_axis = frame
    else:
        up_axis, east_axis, north_axis = np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
        print(" -> Falling back to raw Z-up assumption (no horizontal planes found at all)")


# ---------------------------------------------------------------------------
# 3. Classification -- exact-axis membership instead of raw |normal_z| bands
# ---------------------------------------------------------------------------

def classify_records(records, up_axis):
    floor, ceiling, wall, table, unclassified = [], [], [], [], []
    heights = []
    for r in records:
        proj = np.dot(r.normal, up_axis)
        if abs(proj) > 0.80:
            heights.append((r, np.dot(r.centroid, up_axis)))
        elif abs(proj) < 0.40:
            r.label = "wall"
            wall.append(r)
        else:
            r.label = "unclassified"
            unclassified.append(r)

    if heights:
        heights.sort(key=lambda t: t[1])
        lowest, highest = heights[0][1], heights[-1][1]
        floor_limit, ceiling_limit = lowest + 0.20, highest - 0.20
        for r, h in heights:
            if h <= floor_limit:
                r.label = "floor"
                floor.append(r)
            elif h >= ceiling_limit:
                r.label = "ceiling"
                ceiling.append(r)
            else:
                r.label = "table"
                table.append(r)

    return {"floor": floor, "ceiling": ceiling, "wall": wall, "table": table, "unclassified": unclassified}


print("\nClassifying primary planes into Floor, Ceiling, Walls, and Tables (Manhattan-corrected)...")
classified = classify_records(primary_records, up_axis)
for label in ("floor", "ceiling", "wall", "table"):
    for r in classified[label]:
        print(f" -> Plane {r.id} classified as {label.upper()} "
              f"({'snapped' if r.snapped else 'unsnapped'})")

pcd_non_planar_combined = pcd_non_planar
for r in classified["unclassified"]:
    pcd_non_planar_combined += r.cloud


# ---------------------------------------------------------------------------
# 4. Second-pass bumpy-wall recovery -- same coarse sweep as Map_Cleaning.py
# ---------------------------------------------------------------------------

print("\nRunning Second-Pass Wall Recovery on remaining clutter...")
clutter_working = open3d.geometry.PointCloud(pcd_non_planar_combined)
temp_non_wall = open3d.geometry.PointCloud()
secondary_records = []

for pass_idx in range(4):
    if len(clutter_working.points) < 5000:
        break

    plane_model, inliers = clutter_working.segment_plane(distance_threshold=0.22, ransac_n=3, num_iterations=2000)
    a, b, c, d = plane_model
    normal = np.array([a, b, c])
    normal /= np.linalg.norm(normal)

    plane_pc = clutter_working.select_by_index(inliers)
    clutter_working = clutter_working.select_by_index(inliers, invert=True)

    if abs(normal[2]) < 0.40:                            # coarse clutter sweep; not Manhattan-tight by design
        points = np.asarray(plane_pc.points)
        record = PlaneRecord(
            id=1000 + pass_idx, source="secondary", normal=normal, d=float(d),
            centroid=points.mean(axis=0), inlier_count=len(points), cloud=plane_pc,
            label="wall", snapped=False,
        )
        secondary_records.append(record)
        print(f" -> Recovered bumpy wall plane (columns/windows) from clutter ({len(plane_pc.points)} points)")
    else:
        temp_non_wall += plane_pc

pcd_non_planar = clutter_working + temp_non_wall


# ---------------------------------------------------------------------------
# 5. Plane dedup -- merge walls split by occlusion / secondary-pass recovery
# ---------------------------------------------------------------------------

def merge_planes(records):
    total = sum(r.inlier_count for r in records)
    normal = sum(r.normal * r.inlier_count for r in records) / total
    normal /= np.linalg.norm(normal)
    centroid = sum(r.centroid * r.inlier_count for r in records) / total
    d = float(-np.dot(normal, centroid))
    cloud = records[0].cloud
    for r in records[1:]:
        cloud = cloud + r.cloud
    return PlaneRecord(
        id=-1, source="merged", normal=normal, d=d, centroid=centroid,
        inlier_count=total, cloud=cloud, label=records[0].label,
        snapped=all(r.snapped for r in records),
    )


def dedupe_planes(records, normal_tol_deg=5.0, offset_tol_m=0.15):
    n = len(records)
    if n <= 1:
        return list(records)

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
            cos_a = min(abs(np.dot(records[i].normal, records[j].normal)), 1.0)
            angle = np.degrees(np.arccos(cos_a))
            if angle < normal_tol_deg:
                offset_diff = abs(np.dot(records[i].centroid, records[i].normal)
                                   - np.dot(records[j].centroid, records[i].normal))
                if offset_diff < offset_tol_m:
                    union(i, j)

    groups = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)

    merged = []
    for idxs in groups.values():
        group = [records[i] for i in idxs]
        merged.append(group[0] if len(group) == 1 else merge_planes(group))
    return merged


all_wall_records = classified["wall"] + secondary_records
print(f"\nDeduplicating {len(all_wall_records)} wall plane(s)...")
wall_records = dedupe_planes(all_wall_records, normal_tol_deg=5.0, offset_tol_m=0.15)
print(f" -> {len(wall_records)} distinct wall(s) after dedup")

floor_records = classified["floor"]
ceiling_records = classified["ceiling"]
table_records = classified["table"]


# ---------------------------------------------------------------------------
# 6. Analytic polyhedron: adjacency + plane intersection + mesh assembly
# ---------------------------------------------------------------------------

def intersect_three_planes(p1, p2, p3, det_tol=1e-3):
    A = np.array([p1.normal, p2.normal, p3.normal])
    b = np.array([-p1.d, -p2.d, -p3.d])
    if abs(np.linalg.det(A)) < det_tol:
        return None
    return np.linalg.solve(A, b)


def ring_is_monotonic(angles):
    """True iff the angles (already in candidate ring order) sweep the full
    circle exactly once with no reversal -- i.e. the ring is star-convex
    around its centroid in this order."""
    closed = np.concatenate([angles, angles[:1] + 2 * np.pi])
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


def build_polyhedron(floor_records, ceiling_records, wall_records, up_axis, east_axis, north_axis,
                      parallel_tol_deg=10.0, bbox_slack=0.5):
    if not floor_records or not ceiling_records or len(wall_records) < 3:
        return None, "insufficient planes (need floor, ceiling, and >=3 walls)"

    floor = floor_records[0] if len(floor_records) == 1 else merge_planes(floor_records)
    ceiling = ceiling_records[0] if len(ceiling_records) == 1 else merge_planes(ceiling_records)
    walls = wall_records
    n = len(walls)

    center = np.mean([w.centroid for w in walls], axis=0)

    def angle_of(r):
        rel = r.centroid - center
        return np.arctan2(np.dot(rel, north_axis), np.dot(rel, east_axis))

    walls_sorted = sorted(walls, key=angle_of)

    all_points = np.vstack([np.asarray(r.cloud.points) for r in walls_sorted] +
                            [np.asarray(floor.cloud.points), np.asarray(ceiling.cloud.points)])
    bbox_lo = all_points.min(axis=0) - bbox_slack
    bbox_hi = all_points.max(axis=0) + bbox_slack

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

    def orient(i0, i1, i2, expected_outward):
        v0, v1, v2 = np.array(vertices[i0]), np.array(vertices[i1]), np.array(vertices[i2])
        face_normal = np.cross(v1 - v0, v2 - v0)
        if np.dot(face_normal, expected_outward) < 0:
            return (i0, i2, i1)
        return (i0, i1, i2)

    faces = []
    for i in range(n):
        i1 = (i + 1) % n
        faces.append(orient(idx_c_ceiling, idx_ceiling(i), idx_ceiling(i1), up_axis))
        faces.append(orient(idx_c_floor, idx_floor(i1), idx_floor(i), -up_axis))

        outward = walls_sorted[i].centroid - centroid_floor_3d
        outward = outward - np.dot(outward, up_axis) * up_axis
        norm_outward = np.linalg.norm(outward)
        outward = outward / norm_outward if norm_outward > 1e-6 else walls_sorted[i].normal

        faces.append(orient(idx_floor(i), idx_floor(i1), idx_ceiling(i1), outward))
        faces.append(orient(idx_floor(i), idx_ceiling(i1), idx_ceiling(i), outward))

    mesh = trimesh.Trimesh(vertices=np.array(vertices), faces=np.array(faces), process=True)

    if not mesh.is_watertight:
        return None, "assembled mesh is not watertight"
    if not mesh.is_winding_consistent:
        return None, "assembled mesh winding is inconsistent"
    if mesh.volume <= 0:
        return None, f"non-positive mesh volume ({mesh.volume:.4f}) -- winding bug, not masking with abs()"

    return mesh, None


print("\nBuilding analytic polyhedron from plane equations...")
poly_mesh, poly_fail_reason = build_polyhedron(floor_records, ceiling_records, wall_records,
                                                up_axis, east_axis, north_axis)
if poly_mesh is not None:
    print(f" -> Polyhedron OK: watertight={poly_mesh.is_watertight}, volume={poly_mesh.volume:.3f} m^3")
else:
    print(f" -> Polyhedron unavailable: {poly_fail_reason}")


# ---------------------------------------------------------------------------
# 7. Merge point clouds for saving + convex-hull/footprint fallback
# ---------------------------------------------------------------------------

def merged_cloud(records):
    cloud = open3d.geometry.PointCloud()
    for r in records:
        cloud += r.cloud
    return cloud


floor_pcd = merged_cloud(floor_records)
ceiling_pcd = merged_cloud(ceiling_records)
walls_pcd = merged_cloud(wall_records)
tables_pcd = merged_cloud(table_records)

floor_pcd.paint_uniform_color([0.1, 0.8, 0.1])
ceiling_pcd.paint_uniform_color([0.1, 0.1, 0.8])
walls_pcd.paint_uniform_color([0.8, 0.1, 0.1])
tables_pcd.paint_uniform_color([0.8, 0.8, 0.1])

os.makedirs(output_dir, exist_ok=True)
print(f"\nOutput files will be saved in: {os.path.abspath(output_dir)}")


def save_cloud(filename, pcd):
    filepath = os.path.join(output_dir, filename)
    if not pcd.is_empty():
        open3d.io.write_point_cloud(filepath, pcd)
        print(f" -> Successfully wrote {filepath} ({len(pcd.points)} points)")
    else:
        print(f" -> Skipped {filepath} (Cloud is empty)")


save_cloud("Floor.ply", floor_pcd)
save_cloud("Ceiling.ply", ceiling_pcd)
save_cloud("Walls.ply", walls_pcd)
save_cloud("Tables.ply", tables_pcd)
save_cloud("Clutter_Furniture.ply", pcd_non_planar)


def save_manifest(path, records):
    data = [{
        "id": r.id, "source": r.source, "normal": r.normal.tolist(), "d": r.d,
        "centroid": r.centroid.tolist(), "inlier_count": r.inlier_count,
        "label": r.label, "snapped": r.snapped,
    } for r in records]
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


save_manifest(os.path.join(output_dir, "plane_manifest.json"),
              floor_records + ceiling_records + wall_records + table_records)


def convex_hull_volume(floor_pcd, ceiling_pcd, walls_pcd):
    combined = floor_pcd + ceiling_pcd + walls_pcd
    hull_mesh, _ = combined.compute_convex_hull()
    return hull_mesh.get_volume()


def footprint_height_volume(floor_pcd, ceiling_pcd, walls_pcd, up_axis, east_axis, north_axis):
    wall_points = np.asarray(walls_pcd.points)
    xy = np.column_stack([wall_points @ east_axis, wall_points @ north_axis])
    footprint_area = ConvexHull(xy).volume                 # 2D ConvexHull.volume == enclosed area
    floor_h = np.median(np.asarray(floor_pcd.points) @ up_axis)
    ceiling_h = np.median(np.asarray(ceiling_pcd.points) @ up_axis)
    return footprint_area * (ceiling_h - floor_h)


print("\nComputing cross-check volumes (convex hull, footprint x height)...")
hull_volume = convex_hull_volume(floor_pcd, ceiling_pcd, walls_pcd)
footprint_volume = footprint_height_volume(floor_pcd, ceiling_pcd, walls_pcd, up_axis, east_axis, north_axis)

if poly_mesh is not None:
    primary_volume, primary_method = poly_mesh.volume, "analytic polyhedron"
else:
    primary_volume, primary_method = hull_volume, "convex hull (fallback)"

percent_diff = abs(primary_volume - hull_volume) / primary_volume * 100

print("\n--- Volume Results (Advanced Pipeline) ---")
print(f"Analytic polyhedron volume: {f'{poly_mesh.volume:.2f} m^3' if poly_mesh is not None else 'unavailable (' + poly_fail_reason + ')'}")
print(f"Convex hull volume:         {hull_volume:.2f} m^3")
print(f"Footprint x height volume:  {footprint_volume:.2f} m^3")
print(f"\nPRIMARY VOLUME ({primary_method}): {primary_volume:.2f} m^3")
print(f"Percent difference vs. convex hull: {percent_diff:.1f}%")
print("\nProcessing complete!")
