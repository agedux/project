import open3d as o3d
import numpy as np
from scipy.spatial import cKDTree

# -------------------------------------------------
# Step 1: Load Stanford Bunny
# -------------------------------------------------
try:
    bunny_mesh = o3d.data.BunnyMesh()
    mesh = o3d.io.read_triangle_mesh(bunny_mesh.path)
    print("✅ Loaded Stanford Bunny successfully.")
except Exception as e:
    print(f"⚠️ Could not load BunnyMesh: {e}")
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)

mesh.compute_vertex_normals()

# 原始模型（灰色）
mesh.paint_uniform_color([0.7, 0.7, 0.7])
o3d.visualization.draw_geometries([mesh], window_name="① 原始 Stanford Bunny 模型 (TriangleMesh)")

# -------------------------------------------------
# Step 2: Sample to create point cloud + insert holes
# -------------------------------------------------
pcd_clean = mesh.sample_points_uniformly(number_of_points=20000)
pcd_clean.paint_uniform_color([0.5, 0.5, 0.5])

def insert_eroded_holes(pcd, num_holes=4, radius_ratio=0.15, erosion_factor=0.25):
    """Create visible holes by removing points around random centers."""
    points = np.asarray(pcd.points)
    all_indices_to_remove = []

    bbox_min, bbox_max = pcd.get_min_bound(), pcd.get_max_bound()
    diag_length = np.linalg.norm(bbox_max - bbox_min)

    for _ in range(num_holes):
        hole_center = points[np.random.randint(0, len(points))]
        hole_radius = np.random.uniform(diag_length * radius_ratio * 0.8,
                                        diag_length * radius_ratio * 1.2)
        distances = np.linalg.norm(points - hole_center, axis=1)
        indices_to_remove = np.where(distances < hole_radius)[0]
        num_keep = max(1, int(len(indices_to_remove) * erosion_factor))
        keep_indices = np.random.choice(indices_to_remove, num_keep, replace=False)
        final_remove = np.setdiff1d(indices_to_remove, keep_indices)
        all_indices_to_remove.extend(final_remove)

    points_with_holes = np.delete(points, np.unique(all_indices_to_remove), axis=0)
    pcd_with_holes = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points_with_holes))
    return pcd_with_holes

# 被「掃描」後的兔子點雲（紅色，有洞）
pcd_with_holes = insert_eroded_holes(pcd_clean)
pcd_with_holes.paint_uniform_color([1, 0, 0])
o3d.visualization.draw_geometries([pcd_with_holes], window_name="② 掃描兔子 (有孔洞的點雲)")

# -------------------------------------------------
# Step 3: Local Adaptive α-shape
# -------------------------------------------------
def fill_holes_local_adaptive_alpha(pcd, base_scale=2.0, local_factor=6.0):
    """α-shape reconstruction using locally adaptive α."""
    points = np.asarray(pcd.points)
    if len(points) < 3:
        return pcd

    tree = cKDTree(points)
    dists, _ = tree.query(points, k=8)
    local_mean = np.mean(dists[:, 1:], axis=1)
    global_avg = np.mean(local_mean)

    alpha = np.clip(np.max(local_mean) * base_scale,
                    global_avg * 0.5,
                    global_avg * local_factor)
    print(f"[Alpha-fill] Using adaptive α = {alpha:.5f}")

    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_vertices()
    mesh.compute_vertex_normals()
    return mesh.sample_points_uniformly(len(points))

# -------------------------------------------------
# Step 4: Poisson reconstruction (with normal alignment)
# -------------------------------------------------
def fill_holes_poisson_enhanced(pcd, depth=8):
    """Poisson-based reconstruction with normal alignment and tight cropping."""
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30))
    pcd.orient_normals_consistent_tangent_plane(20)
    mesh_poisson, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)

    bbox = pcd.get_axis_aligned_bounding_box()
    bbox = bbox.scale(1.01, bbox.get_center())  # tighter to prevent bulging
    mesh_crop = mesh_poisson.crop(bbox)
    mesh_crop.remove_degenerate_triangles()
    mesh_crop.remove_duplicated_vertices()
    mesh_crop.compute_vertex_normals()
    return mesh_crop.sample_points_uniformly(len(np.asarray(pcd.points)))

# -------------------------------------------------
# Step 5: Combine α + Poisson
# -------------------------------------------------
def fill_holes_combined(pcd):
    print("Running Local Adaptive Alpha reconstruction ...")
    pcd_alpha = fill_holes_local_adaptive_alpha(pcd)
    print("Running Enhanced Poisson reconstruction ...")
    pcd_poisson = fill_holes_poisson_enhanced(pcd)

    bbox_alpha = pcd_alpha.get_axis_aligned_bounding_box()
    pcd_poisson_cropped = pcd_poisson.crop(bbox_alpha)

    print("Combining α-shape + Poisson results ...")
    combined_points = np.vstack((np.asarray(pcd_alpha.points),
                                 np.asarray(pcd_poisson_cropped.points)))
    pcd_combined = o3d.geometry.PointCloud()
    pcd_combined.points = o3d.utility.Vector3dVector(combined_points)
    pcd_combined = pcd_combined.voxel_down_sample(voxel_size=0.001)
    return pcd_combined

# -------------------------------------------------
# Step 6: Run hole filling & visualize 3 images
# -------------------------------------------------
pcd_filled = fill_holes_combined(pcd_with_holes)
pcd_filled.paint_uniform_color([0, 1, 0])

print("🟢 Repair complete — displaying results...")

# ③ 修補後結果（綠色）
o3d.visualization.draw_geometries(
    [pcd_filled],
    window_name="③ 修補後的兔子 (點雲修補結果)"
)

# -------------------------------------------------
# Optional: Compare all three in one combined scene
# -------------------------------------------------
# 放在不同位置方便觀察
pcd_clean_shift = pcd_clean.translate((-0.05, 0, 0))
pcd_with_holes_shift = pcd_with_holes.translate((0.05, 0, 0))
pcd_filled_shift = pcd_filled.translate((0.15, 0, 0))

o3d.visualization.draw_geometries(
    [pcd_clean_shift, pcd_with_holes_shift, pcd_filled_shift],
    window_name="④ 三種兔子模型對比：原始、掃描、有修補"
)
