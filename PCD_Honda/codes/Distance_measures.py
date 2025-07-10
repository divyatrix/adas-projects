import open3d as o3d
import numpy as np

# Step 1: Load the point cloud
pcd = o3d.io.read_point_cloud("/home/divya1/PCD_Honda/Data/1398110231.025_ascii.pcd")
print("Original point cloud:", pcd)

# Step 2: Apply Translation (Move 3 meters in X, 1 meter up)
pcd.translate((3, 0, 1))
print("Translated the cloud")

# Step 3: Apply Rotation (Rotate 45° around Z axis)
R = pcd.get_rotation_matrix_from_xyz((0, 0, np.pi / 4))
pcd.rotate(R, center=(0, 0, 0))
print("Rotated the cloud")

# Step 4: Apply Scaling (Shrink to 50% size)
pcd.scale(0.5, center=pcd.get_center())
print("Scaled the cloud")

# Step 5: Visualize the transformed point cloud
o3d.visualization.draw_geometries([pcd], window_name="Transformed Cloud")

# Step 6: Save the transformed cloud (optional)
o3d.io.write_point_cloud("/home/divya1/PCD_Honda/Data/transformed_sample.pcd", pcd)
print("Saved transformed point cloud.")