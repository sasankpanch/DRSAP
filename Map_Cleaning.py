# Normal based filtering (maybe)

import open3d
import numpy as np
import torch
import matplotlib.pyplot as plt
import sys, os, urllib.request
import open3d.ml as _ml3d
import open3d.ml.torch as ml3d

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

# Wooooowwwwwww good job :D 
# Wooooowwwwwww good job :D 
# Wooooowwwwwww good job :D 
# 5 Semantic Segmentation
print("Semantic Segmentation..")
CKPT = "randlanet_s3dis.pth"
CFG = "randlanet_s3dis.yml"
CKPT_URL = "https://storage.googleapis.com/open3d-releases/model-zoo/randlanet_s3dis_202201071330utc.pth"
CFG_URL = "https://raw.githubusercontent.com/isl-org/Open3D-ML/main/ml3d/configs/randlanet_s3dis.yml"

KEEP = {0, 1, 2}
CLASS_COLOR = {0: [0.85, 0.85, 0.85], 1: [0.55, 0.35, 0.20], 2: [0.75, 0.70, 0.55]}
out_path = "Room_Semantic.ply"

for f, url in [(CKPT, CKPT_URL), (CFG, CFG_URL)]:
    if not os.path.exists(f):
        print(f"downloading {f} ...")
        urllib.request.urlretrieve(url, f)

pts = np.asarray(pcd_final_clean.points, dtype=np.float32)

if pcd_final_clean.has_colors():
    feat = np.asarray(pcd_final_clean.colors, dtype=np.float32)          # model expects real RGB
else:
    print("pcd_final_clean has no color -- feeding zeros, expect worse results than the S3DIS benchmark")
    feat = np.zeros_like(pts)

cfg = _ml3d.utils.Config.load_from_file(CFG)
model = ml3d.models.RandLANet(**cfg.model)
pipeline = ml3d.pipelines.SemanticSegmentation(model, device="cpu", **cfg.pipeline)
pipeline.load_ckpt(CKPT)

data = {"point" : pts, "feat" : feat, "label" : np.zeros(len(pts), dtype=np.int32)}
labels = pipeline.run_inference(data)["predict_labels"]
mask = np.isin(labels, list(KEEP))

seg = open3d.geometry.PointCloud()
seg.points = open3d.utility.Vector3dVector(pts[mask])
seg.colors = open3d.utility.Vector3dVector(np.array([CLASS_COLOR[l] for l in labels[mask]]))
open3d.io.write_point_cloud(out_path, seg)
print(f"kept {mask.sum()}/{len(pts)} pts (ceiling/floor/wall) -> {out_path}")


#New point cloud based on the previous one but only leaves walls, ceiling and floor
pcd_final_clean = seg

# normal
pcd_final_clean.estimate_normals(search_param=open3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

pcd_final_clean.orient_normals_consistent_tangent_plane(k=15)     # "plane", not "planes"

# 6. RANSAC             
# distance threshold:       how far a point can be away from the plane and count
# normal_angle_threshold:   the degrees of disagreement allowed between point's normal and the plane's normal
# max_planes:               how many planes to lookk for
# min_plane_ratio:          how big the fit has to be to count
print("RANSAC Plane Segmentation")

def plane_segmentor(pcd, max_planes=6, dist_thres=0.03, ransac_n=3, num_iter=1000, min_plane_ratio=0.01, normal_angle_thres=15):
    
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
    max_planes=6, dist_thres=0.03, ransac_n=3, num_iter=1000,
    min_plane_ratio=0.01, normal_angle_thres=15
)

if not plane_clouds:
    raise SystemExit("no planes found -- check that segmentation kept enough structural points")

pcd_structure = plane_clouds[0]
for pc in plane_clouds[1:]:
    pcd_structure += pc

print(f"Yayyyy, Found {len(plane_clouds)} planes and {len(pcd_structure.points)} structural pts, "
      f"{len(pcd_non_planar.points)} were left over as clutter/furniture")

# Outputting files
print("writing files..")
open3d.io.write_point_cloud("Clean_Room.ply", pcd_final_clean)
open3d.io.write_point_cloud("Outlier_Room.ply", pcd_outlier)
open3d.io.write_point_cloud("Room_Structure.ply", pcd_structure)
open3d.io.write_point_cloud("Room_Clutter.ply", pcd_non_planar)