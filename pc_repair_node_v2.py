#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
import open3d as o3d
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
from std_msgs.msg import Header

# ==========================================================
# Stage 1: 離群點移除 — 密度計算
# ==========================================================

def _compute_adaptive_octree_depth(pcd, target_factor=3.0):
    if len(pcd.points) < 10:
        return 3
    diag = np.linalg.norm(pcd.get_max_bound() - pcd.get_min_bound())
    distances = pcd.compute_nearest_neighbor_distance()
    avg_spacing = np.mean(distances)
    if avg_spacing <= 0:
        return 8
    depth = int(np.log2(diag / (avg_spacing * target_factor)))
    return max(3, min(12, depth))


def _get_octree_density_map(pcd, octree_max_depth=-1):
    if len(pcd.points) == 0:
        return np.array([])
    if octree_max_depth <= 0:
        octree_max_depth = _compute_adaptive_octree_depth(pcd)
    octree = o3d.geometry.Octree(max_depth=octree_max_depth)
    octree.convert_from_point_cloud(pcd, size_expand=0.01)
    points = np.asarray(pcd.points)
    leaf_node_key_to_point_indices = {}
    for i, point in enumerate(points):
        leaf_node, info = octree.locate_leaf_node(point)
        if leaf_node:
            node_key = (info.origin[0], info.origin[1], info.origin[2], info.size)
            if node_key not in leaf_node_key_to_point_indices:
                leaf_node_key_to_point_indices[node_key] = []
            leaf_node_key_to_point_indices[node_key].append(i)
    point_densities = np.zeros(len(points))
    for node_key, indices in leaf_node_key_to_point_indices.items():
        density = len(indices)
        for idx in indices:
            point_densities[idx] = density
    point_densities[point_densities == 0] = 1
    return point_densities


# [備用] radius 密度模式 — 待未來測試後啟用
# def _get_radius_density(pcd, radius_multiplier=3.0):
#     if len(pcd.points) < 2:
#         return np.ones(len(pcd.points)) if len(pcd.points) > 0 else np.array([])
#     distances = pcd.compute_nearest_neighbor_distance()
#     avg_spacing = np.mean(distances)
#     radius = avg_spacing * radius_multiplier
#     pcd_tree = o3d.geometry.KDTreeFlann(pcd)
#     points = np.asarray(pcd.points)
#     densities = np.zeros(len(points))
#     for i in range(len(points)):
#         [cnt, _, _] = pcd_tree.search_radius_vector_3d(points[i], radius)
#         densities[i] = cnt
#     densities[densities == 0] = 1
#     return densities


# [備用] Z-score 閾值 — 待未來測試後啟用
# def _zscore_threshold(deltas, n=2.5):
#     return np.mean(deltas) + n * np.std(deltas)


def _mad_threshold(deltas, n=2.5):
    median_delta = np.median(deltas)
    mad = np.median(np.abs(deltas - median_delta))
    return median_delta + n * mad


def remove_outliers(pcd, k=15, threshold_multiplier=2.5,
                    density_octree_depth=-1, min_density=3):
    count_before = len(pcd.points)
    if count_before < k + 1:
        return pcd

    point_densities = _get_octree_density_map(pcd, density_octree_depth)

    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    points = np.asarray(pcd.points)
    deltas = np.zeros(len(points))
    for i in range(len(points)):
        [_, idx, _] = pcd_tree.search_knn_vector_3d(points[i], k + 1)
        neighbors = points[idx[1:]]
        avg_distance = np.mean(np.linalg.norm(points[i] - neighbors, axis=1))
        density = point_densities[i]
        deltas[i] = avg_distance / density if density > 0 else float('inf')

    delta_threshold = _mad_threshold(deltas, threshold_multiplier)

    # δ < 閾值 或 密度夠高（保護表面）
    inlier_indices = np.where(
        (deltas < delta_threshold) | (point_densities >= min_density)
    )[0]
    if len(inlier_indices) < count_before * 0.1:
        return pcd
    return pcd.select_by_index(inlier_indices)

# ==========================================================
# Stage 2: 混淆點移除
# ==========================================================

def remove_confounding_points(pcd, k=20):
    if len(pcd.points) < k:
        return pcd
    points = np.asarray(pcd.points)
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    new_points = np.copy(points)
    for i in range(len(points)):
        [_, idx, _] = pcd_tree.search_knn_vector_3d(points[i], k)
        neighbors = points[idx]
        centroid = np.mean(neighbors, axis=0)
        cov_matrix = np.cov(neighbors.T)
        if np.isnan(cov_matrix).any() or np.isinf(cov_matrix).any():
            continue
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        normal = eigenvectors[:, np.argmin(eigenvalues)]
        projection = points[i] - np.dot(points[i] - centroid, normal) * normal
        new_points[i] = projection
    pcd_projected = o3d.geometry.PointCloud()
    pcd_projected.points = o3d.utility.Vector3dVector(new_points)
    return pcd_projected

# ==========================================================
# Stage 3: 孔洞填補 — BPA
# ==========================================================

def fill_holes_bpa(pcd):
    if len(pcd.points) < 10:
        return pcd
    pcd.estimate_normals()
    pcd.orient_normals_consistent_tangent_plane(k=15)
    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances) if len(distances) > 0 else 0.01
    radii = [avg_dist, avg_dist * 2, avg_dist * 5, avg_dist * 10]
    try:
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd, o3d.utility.DoubleVector(radii))
        if not mesh.has_triangles():
            return pcd
        pcd_filled = mesh.sample_points_uniformly(
            number_of_points=int(len(pcd.points) * 1.1))
        return pcd_filled
    except:
        return pcd

# ==========================================================
# Stage 3: 孔洞填補 — 2D Delaunay 投影法
# ==========================================================

def _triangle_area_from_vertices(a, b, c):
    ab = np.linalg.norm(b - a)
    bc = np.linalg.norm(c - b)
    ca = np.linalg.norm(a - c)
    s = (ab + bc + ca) / 2.0
    area_sq = max(0, s * (s - ab) * (s - bc) * (s - ca))
    return np.sqrt(area_sq)


def fill_holes_delaunay2d(pcd, area_threshold_factor=3.0, max_iterations=50):
    if len(pcd.points) < 10:
        return pcd

    from scipy.spatial import Delaunay

    points = np.asarray(pcd.points)
    centroid = np.mean(points, axis=0)
    centered = points - centroid

    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    basis_2d = eigenvectors[:, 1:]
    points_2d = (centered @ basis_2d)

    try:
        tri = Delaunay(points_2d)
    except:
        return pcd

    triangles = tri.simplices
    areas = np.array([
        _triangle_area_from_vertices(points[t[0]], points[t[1]], points[t[2]])
        for t in triangles
    ])
    area_threshold = np.mean(areas) * area_threshold_factor

    working_points = points.copy()
    working_2d = points_2d.copy()
    holes_filled = 0

    for batch in range(max_iterations):
        tri = Delaunay(working_2d)
        triangles = tri.simplices

        big_tri_verts = []
        for t in triangles:
            area = _triangle_area_from_vertices(
                working_points[t[0]], working_points[t[1]], working_points[t[2]])
            if area > area_threshold:
                big_tri_verts.append(t)

        if len(big_tri_verts) == 0:
            break

        centroids_3d = np.array([
            np.mean(working_points[list(t)], axis=0) for t in big_tri_verts
        ])
        centroids_2d = np.array([
            np.mean(working_2d[list(t)], axis=0) for t in big_tri_verts
        ])

        working_points = np.vstack([working_points, centroids_3d])
        working_2d = np.vstack([working_2d, centroids_2d])
        holes_filled += len(big_tri_verts)

    pcd_filled = o3d.geometry.PointCloud()
    pcd_filled.points = o3d.utility.Vector3dVector(working_points)
    return pcd_filled

# ==========================================================
# ROS 2 Node
# ==========================================================

class PCRepairNode(Node):
    def __init__(self):
        super().__init__('pc_repair_node')

        # Stage 1 params
        self.declare_parameter('outlier_removal_iterations', 0)
        self.declare_parameter('outlier_k', 15)
        self.declare_parameter('outlier_threshold', 2.5)
        self.declare_parameter('density_octree_depth', -1)
        self.declare_parameter('outlier_min_density', 3)

        # Stage 2 params
        self.declare_parameter('confounding_removal_iterations', 0)
        self.declare_parameter('confounding_k', 20)

        # Stage 3 params
        self.declare_parameter('hole_filling_iterations', 0)
        self.declare_parameter('hole_fill_mode', 'bpa')
        self.declare_parameter('hole_area_threshold_factor', 3.0)
        self.declare_parameter('hole_max_iterations', 50)

        # Reset
        self.declare_parameter('reset_cloud', False)

        # Internal state
        self.last_centroid = None
        self.last_count = 0
        self.needs_reprocess = False
        self.force_reprocess = False
        self.current_msg = None

        self.subscription = self.create_subscription(
            PointCloud2, '/object_point_cloud', self.listener_callback, 10)
        self.publisher_ = self.create_publisher(
            PointCloud2, '/repaired_object_pc', 10)
        self.timer = self.create_timer(0.1, self.processing_loop)
        self.add_on_set_parameters_callback(self.parameter_callback)
        self.get_logger().info(
            f'PC Repair Node started (v2). octree+MAD, hole_fill_mode='
            f'{self.get_parameter("hole_fill_mode").value}')

    def parameter_callback(self, params):
        for param in params:
            self.get_logger().info(
                f'Parameter changed: {param.name} = {param.value}')
            if param.name == 'reset_cloud' and param.value is True:
                self.last_centroid = None
                self.last_count = 0
                self.get_logger().info(
                    'RESET: Memory cleared, waiting for next frame.')
        self.force_reprocess = True
        from rcl_interfaces.msg import SetParametersResult
        return SetParametersResult(successful=True)

    def listener_callback(self, msg):
        self.current_msg = msg
        self.needs_reprocess = True

    def processing_loop(self):
        if self.current_msg is None or not self.needs_reprocess:
            return

        step = self.current_msg.point_step
        points_np = np.frombuffer(
            self.current_msg.data, dtype=np.float32).reshape(-1, step // 4)
        points_np = points_np[:, :3]
        mask = np.isfinite(points_np).all(axis=1)
        points_np = points_np[mask]

        if len(points_np) == 0:
            self.needs_reprocess = False
            return

        current_centroid = np.mean(points_np, axis=0)
        current_count = len(points_np)

        is_changed = True
        if self.last_centroid is not None and not self.force_reprocess:
            dist = np.linalg.norm(current_centroid - self.last_centroid)
            count_diff = abs(current_count - self.last_count) / max(
                self.last_count, 1)
            if dist < 0.02 and count_diff < 0.1:
                is_changed = False

        if not is_changed:
            self.needs_reprocess = False
            return

        self.last_centroid = current_centroid
        self.last_count = current_count
        self.force_reprocess = False
        self.needs_reprocess = False

        # Read all params
        n_outlier = self.get_parameter('outlier_removal_iterations').value
        outlier_k = self.get_parameter('outlier_k').value
        threshold = self.get_parameter('outlier_threshold').value
        density_octree_depth = self.get_parameter('density_octree_depth').value
        min_density = self.get_parameter('outlier_min_density').value

        n_smooth = self.get_parameter('confounding_removal_iterations').value
        confounding_k = self.get_parameter('confounding_k').value

        n_hole = self.get_parameter('hole_filling_iterations').value
        hole_fill_mode = self.get_parameter('hole_fill_mode').value
        hole_area_threshold_factor = self.get_parameter(
            'hole_area_threshold_factor').value
        hole_max_iterations = self.get_parameter('hole_max_iterations').value

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_np)
        start_pt_count = len(pcd.points)

        # Stage 1: octree + MAD + density protection
        for _ in range(n_outlier):
            pcd = remove_outliers(
                pcd, k=outlier_k, threshold_multiplier=threshold,
                density_octree_depth=density_octree_depth,
                min_density=min_density)

        # Stage 2
        for _ in range(n_smooth):
            pcd = remove_confounding_points(pcd, k=confounding_k)

        # Stage 3
        for _ in range(n_hole):
            if hole_fill_mode == 'delaunay2d':
                pcd = fill_holes_delaunay2d(
                    pcd, area_threshold_factor=hole_area_threshold_factor,
                    max_iterations=hole_max_iterations)
            else:
                pcd = fill_holes_bpa(pcd)

        end_pt_count = len(pcd.points)
        self.get_logger().info(
            f'REPAIRED: {start_pt_count} -> {end_pt_count} pts '
            f'(hole={hole_fill_mode})')

        repaired_points = np.asarray(pcd.points)
        if len(repaired_points) == 0:
            return
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = self.current_msg.header.frame_id
        repaired_msg = pc2.create_cloud_xyz32(header, repaired_points)
        self.publisher_.publish(repaired_msg)


def main(args=None):
    rclpy.init(args=args)
    node = PCRepairNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
