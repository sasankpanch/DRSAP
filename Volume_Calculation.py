import open3d
import numpy as np
from scipy.spatial import ConvexHull
import os

output_dir = "segmented_output"

# 1. Load segmented clouds
print("Loading segmented point clouds...")
floor_pcd = open3d.io.read_point_cloud(os.path.join(output_dir, "Floor.ply"))
ceiling_pcd = open3d.io.read_point_cloud(os.path.join(output_dir, "Ceiling.ply"))
walls_pcd = open3d.io.read_point_cloud(os.path.join(output_dir, "Walls.ply"))

# 2. Primary approach: 3D convex hull volume
# Tables and Clutter_Furniture are excluded -- they're interior obstructions, not room boundary surfaces
print("Computing convex hull volume...")
combined = floor_pcd + ceiling_pcd + walls_pcd
hull_mesh, _ = combined.compute_convex_hull()

print(f" -> Hull watertight: {hull_mesh.is_watertight()}")   # sanity check, convex hulls are always watertight
hull_volume = hull_mesh.get_volume()

# 3. Fallback approach: robust footprint x height
# median instead of min/max so a single outlier point can't blow out the height estimate
print("Computing footprint x height volume...")
wall_points = np.asarray(walls_pcd.points)
wall_xy = wall_points[:, :2]
footprint_hull = ConvexHull(wall_xy)
footprint_area = footprint_hull.volume     # ConvexHull.volume is the enclosed area for 2D input

floor_z = np.median(np.asarray(floor_pcd.points)[:, 2])
ceiling_z = np.median(np.asarray(ceiling_pcd.points)[:, 2])
room_height = ceiling_z - floor_z

footprint_volume = footprint_area * room_height

# 4. Report both results
print("\n--- Volume Results ---")
print(f"Convex hull volume:        {hull_volume:.2f} m^3")
print(f"Footprint x height volume: {footprint_volume:.2f} m^3")

percent_diff = abs(hull_volume - footprint_volume) / hull_volume * 100
print(f"Percent difference:        {percent_diff:.1f}%")
