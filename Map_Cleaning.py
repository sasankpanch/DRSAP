# Normal based filtering (maybe)

import open3d
import numpy as np
import torch
import matplotlib.pyplot as plt

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using accelerated hardware: {device}")

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

our_mesh = "PLTL-Room-Scan.ply"
baseline_mesh = "PLTL-Room-LIDAR-Scan.ply"
 
print("Loading mesh and Point Cloud")
mesh = open3d.io.read_triangle_mesh(our_mesh)
pcd = mesh.sample_points_uniformly(number_of_points=500000)                         # This is to convert to point cloud


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

labels = np.array(voxel_downsized.cluster_dbscan(eps=eps, min_points=min_points, print_progress= True))

    # extracting largest cluster label
valid_labels = labels[labels >= 0]      

if len(valid_labels) > 0:
    largest_cluster_idx = np.bincount(valid_labels).argmax()
    print(f"Yipeeee!!! Main room structure found, Cluster ID: {largest_cluster_idx}")

    clean_indices = np.where(labels == largest_cluster_idx)[0]
    outlier_indices = np.where(labels != largest_cluster_idx)[0]
    
    pcd_final_clean = voxel_downsized.select_by_index(clean_indices)
    pcd_outlier = voxel_downsized.select_by_index(outlier_indices)

else:
    print("Error, DBSCAN couldn't find distinct clusters, saving downsized cloud instead...")
    pcd_final_clean = voxel_downsized
    pcd_outlier = open3d.geometry.PointCloud()

# 5 Semantic Segmentation
# print("Semantic Segmentation..")
# np.array 

# normal
pcd_final_clean.estimate_normals(search_param=open3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

pcd_final_clean.orient_normals_consistent_tangent_planes(k=15) 

# RANSAC



# open3d.visualization.draw_geometries([pcd_clean])
print("writing files..")
open3d.io.write_point_cloud("Clean_Room.ply", pcd_final_clean)
open3d.io.write_point_cloud("Outlier_Room.ply", pcd_outlier)