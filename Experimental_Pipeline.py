# Normal based filtering (maybe)

import open3d
import numpy as np
import json
import torch
import matplotlib.pyplot as plt
import sys, os, urllib.request
import open3d.ml as _ml3d
import open3d.ml.torch as ml3d

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using accelerated hardware: {device}")

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

our_mesh = "segmented_input/PLTL-Room-Scan.ply"
baseline_mesh = "segmented_input/PLTL-Room-LIDAR-Scan.ply"
conference_mesh = "segmented_input/Conference_Room.ply"
no_cheese_mesh = "segmented_input/No_Cheese.ply"
baseline_editing_room = "segmented_input/Editing-Room-Lidar.ply"
our_editing_room = "segmented_input/Editing-Room.ply"

#file_stat = os.stat(no_cheese_mesh)                                 # <-- change the file
#points = int(file_stat.st_size / 20)
 
print("Loading mesh and Point Cloud")
mesh = open3d.io.read_triangle_mesh(our_mesh)                 # <-- change the file
pcd = mesh.sample_points_uniformly(number_of_points=1000000)                         # This is to convert to point cloud

# 1. Statistical Outlier
print("Statiscal Outlier Removal..")
cl, stat_ind = pcd.remove_statistical_outlier(nb_neighbors = 200, std_ratio=2.0)
pcd_clean = pcd.select_by_index(stat_ind)

# 2. Radius Outlier
print("Radius Outlier Removal..")
cl, rad_ind = pcd_clean.remove_radius_outlier(nb_points=50, radius=0.1)
pcd_clean = pcd_clean.select_by_index(rad_ind)

# 3. Voxel DownSampling
print("Voxel downsampling..")
voxel_size = 0.03
voxel_downsized = pcd_clean.voxel_down_sample(voxel_size = voxel_size)

# 4. DBSCAN Clustering
print("DBSCAN CLustering..")
eps = 0.5                                                                           # nieghborhood radius
min_points = 15         

labels = np.array(voxel_downsized.cluster_dbscan(eps=eps, min_points=min_points, print_progress=True))


valid_indices = np.where(labels >= 0)[0]
outlier_indices = np.where(labels < 0)[0]

if len(valid_indices) > 0:
    print(f"Keeping {len(np.unique(labels[labels >= 0]))} valid clusters.")
    pcd_final_clean = voxel_downsized.select_by_index(valid_indices)
    pcd_outlier = voxel_downsized.select_by_index(outlier_indices)
else:
    print("DBSCAN couldn't find distinct clusters, saving downsized cloud instead...")
    pcd_final_clean = voxel_downsized
    pcd_outlier = open3d.geometry.PointCloud()

# 5. Normalization
pcd_final_clean.estimate_normals(search_param=open3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

pcd_final_clean.orient_normals_consistent_tangent_plane(k=15)     # "plane", not "planes"

# 5.5 Manual Crop (select-to-keep) -- last chance to strip cheese/junk that
# survived every automated filter, right before it can pollute RANSAC's plane
# fits. Uses Open3D's native crop editor: Y,Y to align view, K to lock the
# screen and enter selection mode, drag (rectangle) or ctrl+click (polygon) to
# select the region to KEEP, C to crop, F to leave selection mode. Close the
# window without pressing C to skip/stop.
def manual_crop_loop(pcd):
    current = pcd
    pass_idx = 0
    while True:
        pass_idx += 1
        print(f"\nManual crop pass {pass_idx}: opening viewer window "
              f"({len(current.points)} points currently) -- "
              f"Y,Y to align | K to lock+select | drag or ctrl+click to select region to KEEP | "
              f"C to crop | F to leave selection mode | close window when done.")

        vis = open3d.visualization.VisualizerWithEditing()
        vis.create_window(window_name=f"Manual Crop -- pass {pass_idx} (close when done)")
        vis.add_geometry(current)
        vis.run()
        vis.destroy_window()

        cropped = vis.get_cropped_geometry()
        if cropped is None or len(cropped.points) == 0:
            print(" -> No crop selection made, stopping manual cleanup.")
            break

        print(f" -> Kept {len(cropped.points)} / {len(current.points)} points this pass.")
        current = cropped

        again = input("Run another crop pass? [y/N]: ").strip().lower()
        if again != "y":
            break

    return current

pcd_final_clean = manual_crop_loop(pcd_final_clean)

# 6. RANSAC
# distance threshold:       how far a point can be away from the plane and count
# normal_angle_threshold:   the degrees of disagreement allowed between point's normal and the plane's normal
# max_planes:               how many planes to lookk for
# min_plane_ratio:          how big the fit has to be to count
print("RANSAC Plane Segmentation")

def plane_segmentor(pcd, max_planes=6, dist_thres=0.05, ransac_n=3, num_iter=50000, min_plane_ratio=0.005, normal_angle_thres=25):
    
    min_plane_points = max(100, int(min_plane_ratio * len(pcd.points)))
    remaining = pcd
    plane_models, plane_clouds = [], []

    #loop of the range (RANSAC only does one plane so we run it for the # of planes we need ex: floor, ceeling, walls..)
    for i in range(max_planes):
        #if remaining points is less than min plane points then leave loop
        if len(remaining.points) < min_plane_points:
            break
        
        plane_model, inliers = remaining.segment_plane(distance_threshold=dist_thres, ransac_n=ransac_n,num_iterations=num_iter)
        
        #if length of inliers is less than min plane points then leave loop
        if len(inliers) < min_plane_points:
            break
        
        a, b, c, d = plane_model
        plane_normal = np.array([a, b, c])
        plane_normal /= np.linalg.norm(plane_normal)
        

        inlier_normals = np.asarray(remaining.normals)[inliers]
        cos_angle = np.abs(inlier_normals @ plane_normal)                                   # abs: normal direction can be flipped
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

# RANSAC Function Call
plane_models, plane_clouds, pcd_non_planar = plane_segmentor(
    pcd_final_clean,                                                                    # dropped the stray "pcd"
    max_planes=12, dist_thres=0.05, ransac_n=3, num_iter=10000,
    min_plane_ratio=0.002, normal_angle_thres=25
)

if not plane_clouds:
    raise SystemExit("no planes found -- check that segmentation kept enough structural points")

output_dir = "segmented_output"
os.makedirs(output_dir, exist_ok=True)
print(f"Output files will be saved in: {os.path.abspath(output_dir)}")

# Classifying planes  (band based)
print("\nClassifying primary planes into Floor, Ceiling, Tables, and Walls...")
floor_pcd = open3d.geometry.PointCloud()
ceiling_pcd = open3d.geometry.PointCloud()
walls_pcd = open3d.geometry.PointCloud()
tables_pcd = open3d.geometry.PointCloud()

horizontal_planes_info = []
all_coords = np.asarray(pcd_final_clean.points)
z_midpoint = (np.min(all_coords[:, 2]) + np.max(all_coords[:, 2])) / 2.0

# Per-plane (normal, d, centroid) equations, kept alongside the merged .ply
# clouds so other scripts (e.g. Plane_Mesh_Builder.py) can rebuild geometry
# from the actual RANSAC planes instead of re-running segmentation.
plane_manifest = []

for i, plane_pc in enumerate(plane_clouds):
    a, b, c, d = plane_models[i]
    plane_normal = np.array([a, b, c])
    plane_normal /= np.linalg.norm(plane_normal)

    # Identify Horizontal Surfaces (|n_z| > 0.80)
    if np.abs(plane_normal[2]) > 0.80:
        mean_z = np.mean(np.asarray(plane_pc.points)[:, 2])
        horizontal_planes_info.append((i, mean_z, plane_pc, plane_normal, d))

    # Identify Vertical Surfaces or Walls (|n_z| < 0.40)
    elif np.abs(plane_normal[2]) < 0.40:
        print(f" -> Plane {i} classified as WALL")
        walls_pcd += plane_pc
        points = np.asarray(plane_pc.points)
        plane_manifest.append({
            "id": i, "source": "primary", "label": "wall",
            "normal": plane_normal.tolist(), "d": float(d),
            "centroid": points.mean(axis=0).tolist(), "inlier_count": len(points),
        })
    else:
        pcd_non_planar += plane_pc

# Sorting horizontal planes into height bands
if len(horizontal_planes_info) > 0:
    horizontal_planes_info.sort(key=lambda x: x[1])
    lowest_z = horizontal_planes_info[0][1]
    highest_z = horizontal_planes_info[-1][1]

    floor_band_limit = lowest_z + 0.20
    ceiling_band_limit = highest_z - 0.20

    for idx, mean_z, plane_pc, plane_normal, d in horizontal_planes_info:
        points = np.asarray(plane_pc.points)
        entry = {
            "id": idx, "source": "primary", "normal": plane_normal.tolist(), "d": float(d),
            "centroid": points.mean(axis=0).tolist(), "inlier_count": len(points),
        }
        if mean_z <= floor_band_limit:
            print(f" -> Plane {idx} (z={mean_z:.2f}m) grouped into FLOOR")
            floor_pcd += plane_pc
            entry["label"] = "floor"
        elif mean_z >= ceiling_band_limit:
            print(f" -> Plane {idx} (z={mean_z:.2f}m) grouped into CEILING")
            ceiling_pcd += plane_pc
            entry["label"] = "ceiling"
        else:
            print(f" -> Plane {idx} (z={mean_z:.2f}m) classified as TABLE SURFACE")
            tables_pcd += plane_pc
            entry["label"] = "table"
        plane_manifest.append(entry)

# coarse wall second pass
print("\nRunning Second-Pass Wall Recovery on remaining clutter...")
clutter_working = open3d.geometry.PointCloud(pcd_non_planar)
temp_non_wall = open3d.geometry.PointCloud()

# Run up to 4 coarse passes to extract wide or bumpy vertical structures
for pass_idx in range(4):
    if len(clutter_working.points) < 5000:  # Stop if the leftover cloud is too small
        break
    
    # 22cm threshold allows the plane fit to sweep up column faces and window recesses
    plane_model, inliers = clutter_working.segment_plane(
        distance_threshold=0.22, 
        ransac_n=3,
        num_iterations=2000
    )
    
    a, b, c, d = plane_model
    normal = np.array([a, b, c])
    normal /= np.linalg.norm(normal)
    
    plane_pc = clutter_working.select_by_index(inliers)
    clutter_working = clutter_working.select_by_index(inliers, invert=True)
    
    # If the plane normal is vertical, it's our target wall
    if np.abs(normal[2]) < 0.40:
        print(f" -> Recovered bumpy wall plane (columns/windows) from clutter ({len(plane_pc.points)} points)")
        walls_pcd += plane_pc
        points = np.asarray(plane_pc.points)
        plane_manifest.append({
            "id": 1000 + pass_idx, "source": "secondary", "label": "wall",
            "normal": normal.tolist(), "d": float(d),
            "centroid": points.mean(axis=0).tolist(), "inlier_count": len(points),
        })
    else:
        # If it's a random flat object (like a table top) store it to return to clutter later
        temp_non_wall += plane_pc

# Recombine actual leftover clutter
pcd_non_planar = clutter_working + temp_non_wall

# Recolor elements
floor_pcd.paint_uniform_color([0.1, 0.8, 0.1])      # Green Floor
ceiling_pcd.paint_uniform_color([0.1, 0.1, 0.8])    # Blue Ceiling
walls_pcd.paint_uniform_color([0.8, 0.1, 0.1])      # Red Walls
tables_pcd.paint_uniform_color([0.8, 0.8, 0.1])     # Yellow Tables


# Writing files safely
print("\nWriting files safely to output directory...")

def save_cloud(filename, pcd):
    filepath = os.path.join(output_dir, filename)
    if not pcd.is_empty():
        open3d.io.write_point_cloud(filepath, pcd)
        print(f" -> Successfully wrote {filepath} ({len(pcd.points)} points)")
    else:
        print(f" -> Skipped {filepath} (Cloud is empty)")

save_cloud("Clean_Room.ply", pcd_final_clean)
save_cloud("Floor.ply", floor_pcd)
save_cloud("Ceiling.ply", ceiling_pcd)
save_cloud("Walls.ply", walls_pcd)
save_cloud("Tables.ply", tables_pcd)
save_cloud("Clutter_Furniture.ply", pcd_non_planar)

# 6. Combine the room-shell surfaces into a single cloud
room_combined = floor_pcd + ceiling_pcd + walls_pcd
save_cloud("Room_Combined.ply", room_combined)

manifest_path = os.path.join(output_dir, "plane_manifest.json")
with open(manifest_path, "w") as f:
    json.dump(plane_manifest, f, indent=2)
print(f" -> Successfully wrote {manifest_path} ({len(plane_manifest)} planes)")

print("\nProcessing complete!")