# Normal based filtering (maybe)

import open3d
import numpy as np
import torch
import matplotlib.pyplot as plt
import sys, os, urllib.request
import open3d.ml as _ml3d
import open3d.ml.torch as ml3d

from scipy.ndimage import (binary_closing, binary_dilation, binary_fill_holes,
                           label, minimum_filter1d)
from scipy.spatial import cKDTree

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using accelerated hardware: {device}")

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

our_mesh = "segmented_input/PLTL-Room-Scan.ply"
baseline_mesh = "segmented_input/PLTL-Room-LIDAR-Scan.ply"
conference_mesh = "segmented_input/Conference_Room.ply"
no_cheese_mesh = "segmented_input/No_Cheese.ply"
baseline_editing_room = "segmented_input/Editing-Room-Lidar.ply"
our_editing_room = "segmented_input/Editing-Room.ply"

# ----------------------------------------------------------------------
# Tuning knobs -- everything downstream is derived from these two
# ----------------------------------------------------------------------
voxel_size = 0.03            # metres; sets the density of the whole pipeline
pts_per_voxel = 6            # how many raw samples land in each voxel
plane_tol = 0.04             # floor/ceiling plane thickness
room_height_range = (1.8, 6.0)

output_dir = "segmented_output_concave"
os.makedirs(output_dir, exist_ok=True)
print(f"Output files will be saved in: {os.path.abspath(output_dir)}")


# ======================================================================
# Helpers
# ======================================================================
def find_floor_ceiling(pcd, dist=plane_tol, max_cand=6, h_range=room_height_range):
    """RANSAC only for the two planes it is actually reliable at.

    Floor and ceiling are the largest planes in an indoor scan and they are the
    only pair that is both parallel and separated by a plausible room height,
    so we can identify them without assuming which way is up.
    """
    work = open3d.geometry.PointCloud(pcd)
    n_total = len(pcd.points)
    cands = []

    for _ in range(max_cand):
        if len(work.points) < 0.02 * n_total:
            break
        model, inl = work.segment_plane(dist, 3, 2000)
        if len(inl) < 0.02 * n_total:
            break
        n = np.asarray(model[:3], dtype=float)
        s = np.linalg.norm(n)
        cands.append((n / s, model[3] / s, len(inl)))
        work = work.select_by_index(inl, invert=True)

    best = None
    for i in range(len(cands)):
        for j in range(i + 1, len(cands)):
            n1, d1, c1 = cands[i]
            n2, d2, c2 = cands[j]
            if abs(n1 @ n2) < 0.95:                  # must be parallel
                continue
            d2a = np.sign(n1 @ n2) * d2              # plane 2 in plane 1's frame
            if not h_range[0] <= abs(d1 - d2a) <= h_range[1]:
                continue
            if best is None or c1 + c2 > best[0]:
                best = (c1 + c2, n1, d1, d2a)

    if best is None:
        raise SystemExit("no floor/ceiling pair found -- does the scan cover both?")

    _, n, d1, d2 = best
    up = n * np.sign(n @ np.array([0.0, 0.0, 1.0]))  # IMU frame only breaks the tie
    s = np.sign(n @ up)
    h1, h2 = -d1 * s, -d2 * s
    return up, min(h1, h2), max(h1, h2)


def seeds_from_floor(floor_xy, spacing=0.75, max_seeds=400):
    """Viewpoints for the carve. Floor points are inside the room by definition."""
    if len(floor_xy) == 0:
        raise SystemExit("no floor points to seed the carve from")
    cells = np.floor(floor_xy / spacing).astype(np.int64)
    _, first = np.unique(cells, axis=0, return_index=True)
    seeds = floor_xy[np.sort(first)]
    if len(seeds) > max_seeds:
        rng = np.random.default_rng(0)
        seeds = seeds[rng.choice(len(seeds), max_seeds, replace=False)]
    return seeds


def carve_visible_2d(xy, seeds, n_bins=360, slab=0.15, spread=2):
    """Keep only points that are a first return from at least one interior seed.

    Cheese is welded to the outside of a wall, so it is occluded from every
    point inside the room. `spread` widens each bearing bin's occluder search:
    a wall sampled at one point per voxel is thinner than a bin at several
    metres range, and without it the gaps let everything behind them survive.
    """
    keep = np.zeros(len(xy), dtype=bool)
    bin_scale = n_bins / (2.0 * np.pi)
    width = 2 * spread + 1

    for seed in seeds:
        d = xy - seed
        r = np.hypot(d[:, 0], d[:, 1])
        theta = np.arctan2(d[:, 1], d[:, 0])
        b = ((theta + np.pi) * bin_scale).astype(np.int64) % n_bins

        r_min = np.full(n_bins, np.inf)
        np.minimum.at(r_min, b, r)
        if spread > 0:
            r_min = minimum_filter1d(r_min, width, mode="wrap")

        keep |= r <= r_min[b] + slab

    return keep


def polygon_area(poly):
    """Unsigned shoelace area."""
    poly = np.asarray(poly, dtype=float)
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def room_footprint(xy, seeds, cell, close_gaps=1):
    """Rasterise -> flood fill from the seeds -> fill holes.

    A wall projected onto the floor is a curve a few cm thick, not a filled
    region, so a concave hull on it degenerates into slivers. Rasterising
    dodges that: the fill only has to find the inside, which is unambiguous,
    and it physically cannot cross a closed wall to reach the cheese.
    Furniture becomes a hole in the free space and is closed back up.
    """
    lo = np.minimum(xy.min(axis=0), seeds.min(axis=0)) - 5 * cell
    hi = np.maximum(xy.max(axis=0), seeds.max(axis=0)) + 5 * cell
    shape = tuple(np.ceil((hi - lo) / cell).astype(int) + 1)

    occ = np.zeros(shape, dtype=bool)
    ij = ((xy - lo) / cell).astype(int)
    occ[ij[:, 0], ij[:, 1]] = True

    sij = ((seeds - lo) / cell).astype(int)

    # The grid is padded by 5 cells, so a fill that reaches the border escaped
    # through a hole in the wall. Escalate the closing radius until it doesn't.
    for radius in range(close_gaps, close_gaps + 6):
        walls_grid = binary_closing(occ, np.ones((radius * 2 + 1,) * 2)) if radius else occ
        free_lbl, _ = label(~walls_grid)
        seed_labels = sorted(set(free_lbl[sij[:, 0], sij[:, 1]].tolist()) - {0})
        if not seed_labels:
            raise SystemExit("every seed landed on an occupied cell -- cell size too big?")
        interior = np.isin(free_lbl, seed_labels)
        leaked = (interior[0, :].any() or interior[-1, :].any()
                  or interior[:, 0].any() or interior[:, -1].any())
        if not leaked:
            break

    if leaked:
        raise SystemExit(
            "free space reached the grid border -- the wall cloud has a gap "
            "(open doorway, or a stretch the drone never saw). Close it, or "
            "restrict the seeds to the room you actually want to measure.")

    room = binary_fill_holes(interior | (binary_dilation(interior) & walls_grid))
    area = float(room.sum()) * cell * cell
    return area, trace_polygon(room, lo, cell)


def trace_polygon(mask, origin, cell):
    """Outer contour of the room mask, in metres. None if skimage is missing."""
    try:
        from skimage.measure import find_contours
    except ImportError:
        print(" -> skimage not installed; skipping floorplan polygon")
        return None
    contours = find_contours(mask.astype(float), 0.5)
    if not contours:
        return None
    return max(contours, key=polygon_area) * cell + origin


def simplify_polygon(poly, tol):
    """Douglas-Peucker: stair-stepped raster contour -> straight wall runs."""
    def rdp(pts):
        if len(pts) < 3:
            return pts
        start, end = pts[0], pts[-1]
        seg = end - start
        seg_len = np.linalg.norm(seg)
        if seg_len < 1e-12:
            d = np.linalg.norm(pts - start, axis=1)
        else:
            d = np.abs(np.cross(seg, pts - start)) / seg_len
        i = int(np.argmax(d))
        if d[i] <= tol:
            return np.vstack([start, end])
        return np.vstack([rdp(pts[:i + 1])[:-1], rdp(pts[i:])])

    poly = np.asarray(poly, dtype=float)
    if len(poly) < 3:
        return poly
    return rdp(np.vstack([poly, poly[:1]]))[:-1]


def save_cloud(filename, pcd):
    filepath = os.path.join(output_dir, filename)
    if not pcd.is_empty():
        open3d.io.write_point_cloud(filepath, pcd)
        print(f" -> Successfully wrote {filepath} ({len(pcd.points)} points)")
    else:
        print(f" -> Skipped {filepath} (Cloud is empty)")


# ======================================================================
# 1. Load and sample at a FIXED DENSITY, not a fixed point count
# ======================================================================
print("Loading mesh and Point Cloud")
mesh = open3d.io.read_triangle_mesh(our_mesh)                 # <-- change the file

# sample_points_uniformly is area-weighted, so points / surface area is the
# density we actually control. A big room and a small room now come out at the
# same points per m2, which is what makes every fixed radius below portable.
surface_area = mesh.get_surface_area()
target_density = pts_per_voxel / voxel_size ** 2
n_points = int(np.clip(target_density * surface_area, 200_000, 4_000_000))
print(f"Mesh surface {surface_area:.1f} m2 -> sampling {n_points} points "
      f"({n_points / surface_area:.0f} pts/m2)")
pcd = mesh.sample_points_uniformly(number_of_points=n_points)

# ======================================================================
# 2. Voxel downsample FIRST, then filter
# ======================================================================
print("Voxel downsampling..")
pcd = pcd.voxel_down_sample(voxel_size=voxel_size)

# After voxelising, planar density is 1/voxel^2 no matter how big the room is,
# so the neighbour count is derivable instead of guessed. Expected neighbours
# on a flat patch = density * pi * r^2.
r_out = 0.10
expected = (1.0 / voxel_size ** 2) * np.pi * r_out ** 2
print(f"Radius Outlier Removal.. (expecting ~{expected:.0f} neighbours on a plane)")
pcd, _ = pcd.remove_radius_outlier(nb_points=int(0.35 * expected), radius=r_out)

print("Statiscal Outlier Removal..")
pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=2.0)

# ======================================================================
# 3. DBSCAN -- removes free-floating specks only.
#    Cheese is welded to a wall, so it survives this on purpose.
# ======================================================================
print("DBSCAN CLustering..")
labels = np.array(pcd.cluster_dbscan(eps=0.5, min_points=15, print_progress=True))
valid = np.where(labels >= 0)[0]
if len(valid) > 0:
    print(f"Keeping {len(np.unique(labels[labels >= 0]))} valid clusters.")
    pcd = pcd.select_by_index(valid)
else:
    print("DBSCAN couldn't find distinct clusters, keeping downsized cloud instead...")

# ======================================================================
# 4. Floor and ceiling by RANSAC -> gives us the up axis for free
# ======================================================================
print("\nRANSAC: locating floor and ceiling...")
up, h_floor, h_ceil = find_floor_ceiling(pcd)
height = h_ceil - h_floor
print(f" -> up = {np.round(up, 3)}   floor h={h_floor:.2f}   ceiling h={h_ceil:.2f}"
      f"   height={height:.2f} m")

pts = np.asarray(pcd.points)
h = pts @ up
floor_m = np.abs(h - h_floor) < plane_tol
ceil_m = np.abs(h - h_ceil) < plane_tol
inside = (h > h_floor + plane_tol) & (h < h_ceil - plane_tol)

# height gate: anything stacked above the ceiling or below the floor is cheese
outside_band = ~(floor_m | ceil_m | inside)
print(f" -> height gate dropped {outside_band.sum()} points above/below the room")

# ======================================================================
# 5. Project onto the floor plane and carve from the inside out
# ======================================================================
a = np.array([1.0, 0.0, 0.0]) if abs(up[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
e1 = np.cross(up, a)
e1 /= np.linalg.norm(e1)
e2 = np.cross(up, e1)
to2d = lambda P: np.column_stack([P @ e1, P @ e2])

wall_xy = to2d(pts[inside])
seeds = seeds_from_floor(to2d(pts[floor_m]), spacing=0.75)
print(f"\nCarving from {len(seeds)} interior viewpoints...")
keep = carve_visible_2d(wall_xy, seeds, n_bins=360, slab=0.15, spread=2)
print(f" -> {keep.sum()}/{len(keep)} interior points are visible from inside "
      f"({1 - keep.mean():.1%} carved away as occluded)")

# ======================================================================
# 6. Footprint, floorplan and volume
# ======================================================================
print("\nBuilding footprint...")
area, poly = room_footprint(wall_xy[keep], seeds, cell=2 * voxel_size, close_gaps=1)
print(f"\n  FOOTPRINT {area:.2f} m2   HEIGHT {height:.2f} m   VOLUME {area * height:.2f} m3\n")

if poly is not None:
    corners = simplify_polygon(poly, tol=2 * voxel_size)
    print(f"Floorplan: {len(corners)} corners, {polygon_area(corners):.2f} m2")
    print(np.round(corners, 2).tolist())

# ======================================================================
# 7. Split the survivors and write files
# ======================================================================
print("\nWriting files safely to output directory...")
idx_inside = np.where(inside)[0][keep]

floor_pcd = pcd.select_by_index(np.where(floor_m)[0])
ceiling_pcd = pcd.select_by_index(np.where(ceil_m)[0])

if poly is not None:
    on_shell = cKDTree(poly).query(wall_xy[keep])[0] < 3 * voxel_size
    walls_pcd = pcd.select_by_index(idx_inside[on_shell])
    clutter_pcd = pcd.select_by_index(idx_inside[~on_shell])
else:
    walls_pcd = pcd.select_by_index(idx_inside)
    clutter_pcd = open3d.geometry.PointCloud()

floor_pcd.paint_uniform_color([0.1, 0.8, 0.1])      # Green Floor
ceiling_pcd.paint_uniform_color([0.1, 0.1, 0.8])    # Blue Ceiling
walls_pcd.paint_uniform_color([0.8, 0.1, 0.1])      # Red Walls
clutter_pcd.paint_uniform_color([0.8, 0.8, 0.1])    # Yellow Clutter

save_cloud("Clean_Room.ply", floor_pcd + ceiling_pcd + walls_pcd + clutter_pcd)
save_cloud("Floor.ply", floor_pcd)
save_cloud("Ceiling.ply", ceiling_pcd)
save_cloud("Walls.ply", walls_pcd)
save_cloud("Clutter_Furniture.ply", clutter_pcd)
save_cloud("Room_Combined.ply", floor_pcd + ceiling_pcd + walls_pcd)

if poly is not None:
    fp = open3d.geometry.PointCloud()
    fp.points = open3d.utility.Vector3dVector(
        np.array([p[0] * e1 + p[1] * e2 + h_floor * up for p in poly]))
    fp.paint_uniform_color([1.0, 0.0, 1.0])
    save_cloud("Footprint.ply", fp)

print("\nProcessing complete!")