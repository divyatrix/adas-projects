import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

# ---------- Load & Visualize ----------
pcd = o3d.io.read_point_cloud("sample.pcd")
print("Original Point Cloud:", pcd)
o3d.visualization.draw_geometries([pcd], window_name="Original Point Cloud")

# ---------- Cleaning ----------
pcd.remove_non_finite_points()
points = np.asarray(pcd.points)
print("Stats - Min:", points.min(axis=0), "Max:", points.max(axis=0), "Mean:", points.mean(axis=0))

# ---------- Filtering ----------
pcd, _ = pcd.remove_radius_outlier(nb_points=5, radius=0.1)
pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
o3d.visualization.draw_geometries([pcd], window_name="Filtered Point Cloud")

# ---------- Downsampling ----------
voxel_down = pcd.voxel_down_sample(voxel_size=0.05)
uniform_down = pcd.uniform_down_sample(every_k_points=10)

# ---------- Estimate Normals ----------
voxel_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
o3d.visualization.draw_geometries([voxel_down], point_show_normal=True)

# ---------- Transformations ----------
pcd.translate((2, 0, 0))
R = pcd.get_rotation_matrix_from_xyz((0, 0, np.pi / 4))
pcd.rotate(R, center=(0, 0, 0))
pcd.scale(0.5, center=pcd.get_center())

# ---------- Bounding Boxes ----------
aabb = pcd.get_axis_aligned_bounding_box()
obb = pcd.get_oriented_bounding_box()
aabb.color = (1, 0, 0)
obb.color = (0, 1, 0)
o3d.visualization.draw_geometries([pcd, aabb, obb])

# ---------- DBSCAN Clustering ----------
labels = np.array(pcd.cluster_dbscan(eps=0.2, min_points=10, print_progress=True))
max_label = labels.max()
colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
colors[labels < 0] = [0, 0, 0]
pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
o3d.visualization.draw_geometries([pcd], window_name="DBSCAN Clustering")

# ---------- Cluster Bounding Boxes ----------
cluster_indices = np.where(labels == 0)[0]
cluster = pcd.select_by_index(cluster_indices)
aabb = cluster.get_axis_aligned_bounding_box()
obb = cluster.get_oriented_bounding_box()
aabb.color = (1, 0, 0)
obb.color = (0, 1, 0)
o3d.visualization.draw_geometries([cluster, aabb, obb])

# ---------- Distance Calculations ----------
dist = np.linalg.norm(points[0] - points[10])
print("Distance between point[0] and point[10]:", dist)

plane_model, inliers = pcd.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=1000)
[a, b, c, d] = plane_model
x, y, z = points[0]
dist_to_plane = abs(a*x + b*y + c*z + d) / np.sqrt(a**2 + b**2 + c**2)
print("Distance of point[0] to plane:", dist_to_plane)

# ---------- Preprocessing for Deep Learning ----------
center = np.mean(points, axis=0)
pcd.translate(-center)
scale = np.max(np.linalg.norm(points, axis=1))
points /= scale
pcd.points = o3d.utility.Vector3dVector(points)

np.save("cloud_1024.npy", np.asarray(pcd.points)[:1024])  # Save first 1024 points
o3d.io.write_point_cloud("processed_output.pcd", pcd)
