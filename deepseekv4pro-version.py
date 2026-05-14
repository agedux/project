import open3d as o3d
import numpy as np
import os
from datetime import datetime
from scipy.spatial import cKDTree

def add_defects(pcd, noise_level_percentage, num_holes=5):
    """
    在點雲上模擬並添加瑕疵，包括噪點、離群點和孔洞。

    Args:
        pcd (open3d.geometry.PointCloud): 原始的乾淨點雲。
        noise_level_percentage (float): 噪點等級百分比，用於控制噪點的數量與強度。
        num_holes (int): 要在點雲上製造的孔洞數量。

    Returns:
        tuple:
            - open3d.geometry.PointCloud: 含有瑕疵的新點雲。
            - numpy.ndarray: 所有被視為「真實噪點」的點的索引，用於後續評估去噪演算法的性能。
    """
    noisy_pcd = o3d.geometry.PointCloud(pcd)
    points = np.asarray(noisy_pcd.points)
    num_points_clean = len(points)

    # --- 步驟 1: 添加噪點 (高斯噪點與隨機離群點) ---
    # 根據點雲的邊界框大小計算噪點的強度
    noise_magnitude = (noise_level_percentage / 100.0) * np.mean(pcd.get_max_bound() - pcd.get_min_bound()) / 5
    # 根據噪點比例計算要添加高斯噪點的點數
    num_noisy_points = int(num_points_clean * (noise_level_percentage / 100.0))
    
    noise_indices_gaussian = np.array([], dtype=int)
    if num_noisy_points > 0:
        # 隨機選取點，並加上符合高斯分佈的位移，模擬緊貼表面的「混淆點」
        noisy_indices_gaussian = np.random.choice(num_points_clean, num_noisy_points, replace=False)
        noise = np.random.normal(0, noise_magnitude, (num_noisy_points, 3))
        points[noisy_indices_gaussian] += noise

    # 添加完全隨機的「離群點」
    num_random_outliers = int(num_points_clean * (noise_level_percentage / 100.0 / 2.0))
    if num_random_outliers > 0:
        min_bound = pcd.get_min_bound()
        max_bound = pcd.get_max_bound()
        # 在點雲的邊界框內生成均勻分佈的隨機點
        random_outliers = np.random.uniform(min_bound, max_bound, (num_random_outliers, 3))
        noise_indices_random = np.arange(len(points), len(points) + len(random_outliers))
        points = np.vstack((points, random_outliers))
    else:
        noise_indices_random = np.array([], dtype=int)

    # Stage 1 應移除的只有隨機離群點（Gaussian 擾動點留給 Stage 2 修正）
    ground_truth_noise_indices = noise_indices_random.copy()
    
    noisy_pcd.points = o3d.utility.Vector3dVector(points)

    # --- 步驟 2: 製造孔洞 ---
    if num_holes > 0 and len(np.asarray(noisy_pcd.points)) > 0:
        points_for_holes = np.asarray(noisy_pcd.points)
        all_indices_to_remove = []
        
        # 使用點雲的邊界框對角線長度來標準化孔洞半徑
        bbox_min, bbox_max = noisy_pcd.get_min_bound(), noisy_pcd.get_max_bound()
        diag_length = np.linalg.norm(bbox_max - bbox_min)
        # 半徑比例，可根據需求調整
        radius_ratio = (noise_level_percentage / 100.0) * 0.2 
        hole_radius = diag_length * radius_ratio
        # 侵蝕因子，值越小，孔洞越明顯
        erosion_factor = 0.2

        for _ in range(num_holes):
            if len(points_for_holes) == 0: break
            # 隨機選取一個點作為孔洞中心
            hole_center = points_for_holes[np.random.randint(0, len(points_for_holes))]
            
            # 計算所有點到中心的距離
            distances = np.linalg.norm(points_for_holes - hole_center, axis=1)
            
            # 找到半徑內的所有點
            indices_in_radius = np.where(distances < hole_radius)[0]
            if len(indices_in_radius) < 10: # 如果區域內點太少，就跳過，避免產生微小孔洞
                continue

            # 從半徑內的點中，隨機保留一部分 (侵蝕效果)
            num_keep = max(1, int(len(indices_in_radius) * erosion_factor))
            keep_indices = np.random.choice(indices_in_radius, num_keep, replace=False)
            
            # 確定最終要移除的點 (半徑內的點 減去 要保留的點)
            final_remove_indices = np.setdiff1d(indices_in_radius, keep_indices)
            all_indices_to_remove.extend(final_remove_indices)

        # 從點雲中刪除所有被標記為移除的點
        unique_indices_to_remove = np.unique(all_indices_to_remove)
        
        # 創建一個遮罩，標記要被移除的點
        points_to_keep_mask = np.ones(len(points_for_holes), dtype=bool)
        points_to_keep_mask[unique_indices_to_remove] = False
        
        # 根據遮罩選取要保留的點，從而移除孔洞區域的點
        noisy_pcd = noisy_pcd.select_by_index(np.where(points_to_keep_mask)[0])
    
    return noisy_pcd, ground_truth_noise_indices

def calculate_denoising_metrics(total_points_in_noisy_pcd, ground_truth_noise_indices, denoised_inlier_indices):
    """
    計算並印出去噪演算法的性能指標 (Precision, Recall, F1-score)。

    Args:
        total_points_in_noisy_pcd (int): 帶噪點雲的總點數。
        ground_truth_noise_indices (set): 真實噪點的索引集合。
        denoised_inlier_indices (set): 經過演算法處理後，被判定為「內點」(Inlier) 的索引集合。
        
    在點雲去噪的 F-score 計算裡：
        TP (True Positive)
        正確砍掉的離群點 ✓
        （你砍了，它確實是垃圾點）

        FP (False Positive)
        誤砍的物體表面點 ✗
        （你砍了，但它其實是好的點）

        FN (False Negative)  
        漏砍的離群點 ✗
        （垃圾點還在，你沒砍到）

    你的狀況：

    MAD threshold=2.5:  砍很多 → 離群點幾乎全砍 ✓（FN 低）
        → 但也砍到一些表面點 ✗（FP 高）
        → Precision = TP/(TP+FP) 低
        → Recall 高，Precision 低

    MAD threshold=5~8: 砍得少 → 只砍最明顯的垃圾
        → FP 低 → Precision 高
        → 可能漏一些邊緣 outlier → FN 略高
        → 兩者平衡 → F1 高
    """
    all_indices = set(range(total_points_in_noisy_pcd))
    ground_truth_noise_set = set(ground_truth_noise_indices)
    # 真實的乾淨點 = 所有點 - 真實的噪點
    ground_truth_clean_set = all_indices - ground_truth_noise_set

    # 被演算法移除的點 = 所有點 - 演算法判定的內點
    removed_indices = all_indices - set(denoised_inlier_indices)

    # 真陽性 (TP): 被正確移除的噪點 (演算法移除的點 與 真實噪點 的交集)
    tp = len(removed_indices.intersection(ground_truth_noise_set))
    
    # 偽陽性 (FP): 被錯誤移除的乾淨點 (演算法移除的點 與 真實乾淨點 的交集)
    fp = len(removed_indices.intersection(ground_truth_clean_set))

    # 偽陰性 (FN): 未被移除的噪點 (真實噪點的總數 - 被正確移除的噪點)
    fn = len(ground_truth_noise_set) - tp

    # 根據公式計算指標
    # 精確率 (Precision, Pd): 在所有被移除的點中，有多少比例是真正的噪點
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    # 召回率 (Recall, Rd): 在所有真正的噪點中，有多少比例被成功移除了
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    # F1-Score: 精確率和召回率的調和平均數，是綜合評價指標
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    print("--- 去噪性能指標 ---")
    print(f"    精確率 (Precision, Pd): {precision:.4f}")
    print(f"    召回率 (Recall, Rd):    {recall:.4f}")
    print(f"    F1-score:               {f1_score:.4f}")
    print("----------------------------------------")

# ==========================================================
# Stage 1: 離群點移除 — 密度計算
# ==========================================================

def _compute_adaptive_octree_depth(pcd, target_factor=3.0):
    """根據點雲尺度自適應計算 octree 深度。

    leaf voxel 邊長 ≈ avg_spacing × target_factor
    """
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
    """自適應八叉樹密度。depth=-1 時自動計算。"""
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


def _get_radius_density(pcd, radius_multiplier=3.0):
    """半徑密度：每個點在固定半徑內的鄰居數。"""
    if len(pcd.points) < 2:
        return np.ones(len(pcd.points)) if len(pcd.points) > 0 else np.array([])
    distances = pcd.compute_nearest_neighbor_distance()
    avg_spacing = np.mean(distances)
    radius = avg_spacing * radius_multiplier

    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    points = np.asarray(pcd.points)
    densities = np.zeros(len(points))
    for i in range(len(points)):
        [cnt, _, _] = pcd_tree.search_radius_vector_3d(points[i], radius)
        densities[i] = cnt
    densities[densities == 0] = 1
    return densities


# ==========================================================
# Stage 1: 離群點移除 — 閾值計算
# ==========================================================

def _zscore_threshold(deltas, n=2.5):
    """Z-score 閾值。"""
    return np.mean(deltas) + n * np.std(deltas)


def _mad_threshold(deltas, n=2.5):
    """MAD (Median Absolute Deviation) 閾值 — 穩健統計量，不受極端 outlier 汙染。"""
    median_delta = np.median(deltas)
    mad = np.median(np.abs(deltas - median_delta))
    return median_delta + n * mad


def remove_outliers(pcd, k=15, threshold_multiplier=2.5,
                    density_mode="octree", threshold_method="mad",
                    density_octree_depth=-1, density_radius_multiplier=3.0):
    """移除離群點。論文 δ = avg_knn_distance / ρn。

    Args:
        density_mode: "octree" (自適應深度) 或 "radius" (固定半徑)
        threshold_method: "mad" (穩健) 或 "zscore"
        density_octree_depth: -1 自動計算，正整數手動指定
    """
    if len(pcd.points) < k + 1:
        return pcd, np.arange(len(pcd.points))

    # 1. 密度
    if density_mode == "radius":
        point_densities = _get_radius_density(pcd, density_radius_multiplier)
    else:
        point_densities = _get_octree_density_map(pcd, density_octree_depth)

    # 2. δ = avg_knn_distance / ρn
    points = np.asarray(pcd.points)
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    deltas = np.zeros(len(points))
    for i in range(len(points)):
        [_, idx, _] = pcd_tree.search_knn_vector_3d(points[i], k + 1)
        neighbors = points[idx[1:]]
        avg_distance = np.mean(np.linalg.norm(points[i] - neighbors, axis=1))
        density = point_densities[i]
        deltas[i] = avg_distance / density if density > 0 else float('inf')

    if len(deltas) == 0:
        return pcd, np.arange(len(pcd.points))

    # 3. 閾值
    if threshold_method == "zscore":
        delta_threshold = _zscore_threshold(deltas, threshold_multiplier)
    else:
        delta_threshold = _mad_threshold(deltas, threshold_multiplier)

    # 4. 安全檢查：存活點數不可少於 10%
    inlier_indices = np.where(deltas < delta_threshold)[0]
    if len(inlier_indices) < len(points) * 0.1:
        return pcd, np.arange(len(pcd.points))

    return pcd.select_by_index(inlier_indices), inlier_indices

def remove_confounding_points(pcd, k=20):
    """移除混淆點 (平滑化)。PCA 局部平面擬合與投影。

    Args:
        k: 局部平面擬合的鄰居數。小=保留細節，大=平滑強。
    """
    if len(pcd.points) < k:
        return pcd
    points = np.asarray(pcd.points)
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    new_points = np.copy(points)

    # 遍歷每個點
    for i in range(len(points)):
        # 步驟 1: 搜尋 k 個最近的鄰居
        [_, idx, _] = pcd_tree.search_knn_vector_3d(points[i], k)
        neighbors = points[idx]

        # 步驟 2: 使用鄰居點進行局部平面擬合
        # 計算鄰居點的質心
        centroid = np.mean(neighbors, axis=0)
        # 計算協方差矩陣
        cov_matrix = np.cov(neighbors.T)
        if np.isnan(cov_matrix).any() or np.isinf(cov_matrix).any():
            continue
        # 透過計算協方差矩陣的特徵向量，找到最小特徵值對應的特徵向量，即為平面的法向量
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        normal = eigenvectors[:, np.argmin(eigenvalues)]

        # 步驟 3: 將當前點投影到擬合出的局部平面上
        projection = points[i] - np.dot(points[i] - centroid, normal) * normal
        new_points[i] = projection
        
    pcd_projected = o3d.geometry.PointCloud()
    pcd_projected.points = o3d.utility.Vector3dVector(new_points)
    return pcd_projected

# ==========================================================
# Stage 3: 孔洞填補 — 2D Delaunay 投影法
# ==========================================================

def _triangle_area_from_vertices(a, b, c):
    """Heron 公式計算 3D 三角形面積。"""
    ab = np.linalg.norm(b - a)
    bc = np.linalg.norm(c - b)
    ca = np.linalg.norm(a - c)
    s = (ab + bc + ca) / 2.0
    area_sq = max(0, s * (s - ab) * (s - bc) * (s - ca))
    return np.sqrt(area_sq)


def fill_holes_delaunay2d(pcd, area_threshold_factor=3.0, max_iterations=100):
    """2D Delaunay 投影法填補孔洞。

    論文精神對照：
    - 三角化 → 2D Delaunay（取代 alpha-shape，不會跳過大洞）
    - 大三角形 = 破孔 → Heron 公式面積
    - 最大面積三角形重心插點 → 迭代

    適用於 2.5D 點雲（eye-to-hand 桌面場景）。
    """
    if len(pcd.points) < 10:
        return pcd

    points = np.asarray(pcd.points)
    centroid = np.mean(points, axis=0)
    centered = points - centroid

    # 1. PCA → 2D 投影基底
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    basis_2d = eigenvectors[:, 1:]  # (3, 2) — 最大兩個 principal component
    points_2d = (centered @ basis_2d)  # shape (N, 2)

    # 2. 計算平均三角形面積當基準
    from scipy.spatial import Delaunay
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

    # 3. 批次迭代：一次找出所有大三角形 → 一口氣插點 → 重三角化
    working_points = points.copy()
    working_2d = points_2d.copy()
    holes_filled = 0

    for batch in range(max_iterations):
        tri = Delaunay(working_2d)
        triangles = tri.simplices

        # 收集所有超過閾值的三角形
        big_tri_verts = []
        for t in triangles:
            area = _triangle_area_from_vertices(
                working_points[t[0]], working_points[t[1]], working_points[t[2]])
            if area > area_threshold:
                big_tri_verts.append(t)

        if len(big_tri_verts) == 0:
            break

        # 一次插入所有重心點
        centroids_3d = np.array([np.mean(working_points[list(t)], axis=0) for t in big_tri_verts])
        centroids_2d = np.array([np.mean(working_2d[list(t)], axis=0) for t in big_tri_verts])

        working_points = np.vstack([working_points, centroids_3d])
        working_2d = np.vstack([working_2d, centroids_2d])
        holes_filled += len(big_tri_verts)

    print(f"  2D Delaunay 補洞: {holes_filled} 點, {batch+1} 批次 "
          f"(閾值={area_threshold_factor}×avg, {len(working_points)} pts)")

    pcd_filled = o3d.geometry.PointCloud()
    pcd_filled.points = o3d.utility.Vector3dVector(working_points)
    return pcd_filled

def main():
    # --- 1. 設置與載入資料 ---
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # --- 載入資料 ---
    # 請在此處設置要載入的點雲檔案路徑
    file_path = "rawdata/bunny.ply" 
    
    if not os.path.exists(file_path):
        print(f"錯誤: 檔案 '{file_path}' 不存在。")
        print("請檢查檔案路徑是否正確。")
        # 如果檔案不存在，創建一個預設的球體以繼續執行
        print("將創建一個球體作為替代。")
        mesh = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
        pcd_clean = mesh.sample_points_uniformly(number_of_points=20000)
        data_name = "sphere_fallback"
    else:
        # 從檔名中自動獲取 data_name，用於後續儲存檔案
        data_name = os.path.splitext(os.path.basename(file_path))[0]
        print(f"正在從 '{file_path}' 載入檔案...")
        try:
            file_extension = os.path.splitext(file_path)[1].lower()
            
            if file_extension == ".obj":
                print("  偵測到 OBJ 檔案，將其作為網格讀取並提取頂點。")
                mesh = o3d.io.read_triangle_mesh(file_path)
                if not mesh.has_vertices():
                    raise ValueError("OBJ 檔案中沒有頂點。")
                pcd_clean = o3d.geometry.PointCloud()
                pcd_clean.points = mesh.vertices
                # 如果網格有頂點顏色，也一併複製過來
                if mesh.has_vertex_colors():
                    pcd_clean.colors = mesh.vertex_colors
            else:
                # 預設為讀取點雲格式 (PLY, PCD, XYZ 等)
                print("  偵測到點雲檔案格式，直接讀取。")
                pcd_clean = o3d.io.read_point_cloud(file_path)

            if not pcd_clean.has_points():
                # 在檔案格式錯誤或為空時，read_point_cloud/read_triangle_mesh 可能返回一個空物件
                raise ValueError("檔案格式錯誤、檔案為空，或無法提取點。")
            
            print(f"成功載入 {len(pcd_clean.points)} 個點。")
        except Exception as e:
            print(f"從 '{file_path}' 載入時發生錯誤: {e}")
            # 如果載入失敗，創建一個預設的球體以繼續執行
            print("將創建一個球體作為替代。")
            mesh = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
            pcd_clean = mesh.sample_points_uniformly(number_of_points=20000)
            data_name = "sphere_fallback"

    pcd_clean.paint_uniform_color([0.5, 0.5, 0.5]) # 灰色
    print("顯示原始乾淨點雲...")
    o3d.visualization.draw_geometries([pcd_clean], window_name="Original Clean Point Cloud")
    ''''''
    # --- 2. 添加瑕疵 (噪點與孔洞) ---
    noise_levels = [20] # 專注於一個噪點等級進行互動式調參

    for noise_level in noise_levels:
        print(f"\n========== 使用 {noise_level}% 噪點等級進行處理 ========== ")
        
        pcd_clean_copy = o3d.geometry.PointCloud(pcd_clean)

        # 呼叫 add_defects 函式生成帶有瑕疵的點雲
        pcd_noisy, noise_indices = add_defects(pcd_clean_copy, noise_level, num_holes=7)
        pcd_noisy.paint_uniform_color([1, 0, 0]) # 紅色
        print(f"  添加瑕疵後的點數: {len(pcd_noisy.points)} (包含 {len(noise_indices)} 個噪點)")
        
        # 視覺化並截圖 (增加噪點)
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=f"{noise_level}% Noisy Point Cloud", width=1280, height=720)
        vis.add_geometry(pcd_noisy)
        vis.run()
        # 截圖邏輯
        save_path = os.path.join("visualizations", "增加噪點", f"{data_name}_noisy_{datetime.now().strftime('%Y%m%d-%H%M%S')}.png")
        if not os.path.exists(os.path.dirname(save_path)): os.makedirs(os.path.dirname(save_path))
        vis.capture_screen_image(save_path)
        print(f"  [視覺化截圖] 已儲存至: {save_path}")
        vis.destroy_window()
        
        # --- 3. 執行點雲優化流程 ---
        
        # --- 步驟 3.1: 移除離群點 (Stage 1: 自適應 octree + MAD) ---
        print("1. Stage 1: 移除離群點 (adaptive octree + MAD)...")
        print(f"   density_mode=octree(auto), threshold_method=mad, k=15, threshold=2.5")
        pcd_outliers_removed, inlier_indices_denoised = remove_outliers(
            pcd_noisy, k=15, threshold_multiplier=6,
            density_mode="octree", threshold_method="mad")

        # --- Stage 1 四模式比較 (視覺化 + F-score) ---
        # 注意：這裡只評估 Stage 1 的離群點移除效果
        # 完整量化（Stage 1+2）放在混淆點移除之後
        print("\n--- Stage 1 四模式比較 (僅離群點移除) ---")
        combos = [
            ("octree", "mad"),
            ("octree", "zscore"),
            ("radius", "mad"),
            ("radius", "zscore"),
        ]
        results = {}
        for dm, tm in combos:
            pcd_result, idx = remove_outliers(pcd_noisy, k=15, threshold_multiplier=2.5,
                                               density_mode=dm, threshold_method=tm)
            results[(dm, tm)] = pcd_result
            all_idx = set(range(len(pcd_noisy.points)))
            gt_noise = set(noise_indices)
            gt_clean = all_idx - gt_noise
            removed = all_idx - set(idx)
            tp = len(removed & gt_noise)
            fp = len(removed & gt_clean)
            fn = len(gt_noise) - tp
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            print(f"  {dm:>7} + {tm:>6}  →  Pd={precision:.4f}  Rd={recall:.4f}  F1={f1:.4f}")
        print("")

        # 並排視覺化：Noisy + 4 種結果
        span = pcd_noisy.get_max_bound()[0] - pcd_noisy.get_min_bound()[0]
        offset_x = span * 1.2

        pcd_noisy_shift = o3d.geometry.PointCloud(pcd_noisy)
        pcd_noisy_shift.paint_uniform_color([1, 0, 0])  # 紅 = 原始 noisy

        geometries = [pcd_noisy_shift]
        colors = [[0, 1, 0], [0, 0, 1], [1, 1, 0], [0, 1, 1]]  # 綠藍黃青
        labels = ["octree+mad", "octree+zscore", "radius+mad", "radius+zscore"]

        for idx_combo, (dm, tm) in enumerate(combos):
            pcd_shifted = o3d.geometry.PointCloud(results[(dm, tm)])
            pcd_shifted.paint_uniform_color(colors[idx_combo])
            pcd_shifted.translate((offset_x * (idx_combo + 1), 0, 0))
            geometries.append(pcd_shifted)
            print(f"  [{labels[idx_combo]}] 點數: {len(results[(dm, tm)].points)}")

        vis = o3d.visualization.Visualizer()
        vis.create_window(
            window_name="Stage 1 Comparison: Red=Noisy | Green=octree+mad | Blue=octree+zscore | Yellow=radius+mad | Cyan=radius+zscore",
            width=1600, height=720)
        for g in geometries:
            vis.add_geometry(g)
        vis.run()
        save_path = os.path.join("visualizations", "移除離散點",
            f"{data_name}_stage1_comparison_{datetime.now().strftime('%Y%m%d-%H%M%S')}.png")
        if not os.path.exists(os.path.dirname(save_path)): os.makedirs(os.path.dirname(save_path))
        vis.capture_screen_image(save_path)
        print(f"  [並排比較截圖] 已儲存至: {save_path}")
        vis.destroy_window()

        pcd_outliers_removed.paint_uniform_color([0, 1, 0]) # 綠色
        
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = os.path.join(output_dir, f"{data_name}_outliers_removed_{timestamp}.ply")
        o3d.io.write_point_cloud(filename, pcd_outliers_removed)
        print(f"  已儲存: {filename}")

        # 視覺化並截圖 (移除離散點)
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="1. Outliers Removed", width=1280, height=720)
        vis.add_geometry(pcd_outliers_removed)
        vis.run()
        # 截圖邏輯
        save_path = os.path.join("visualizations", "移除離散點", f"{data_name}_outliers_removed_{datetime.now().strftime('%Y%m%d-%H%M%S')}.png")
        if not os.path.exists(os.path.dirname(save_path)): os.makedirs(os.path.dirname(save_path))
        vis.capture_screen_image(save_path)
        print(f"  [視覺化截圖] 已儲存至: {save_path}")
        vis.destroy_window()

        # --- 步驟 3.2: 移除混淆點 (表面平滑化) ---
        print("2. 正在移除混淆點 (局部平面投影)...")
        pcd_confounding_removed = pcd_outliers_removed
        smoothing_iterations = 3 
        smoothing_k = 30 
        for i in range(smoothing_iterations):
            print(f"  平滑化迭代 {i+1}/{smoothing_iterations}...")
            pcd_confounding_removed = remove_confounding_points(pcd_confounding_removed, k=smoothing_k)

        pcd_confounding_removed.paint_uniform_color([0, 0, 1]) # 藍色
        print(f"  所有平滑步驟後的點數: {len(pcd_confounding_removed.points)}")

        # Stage 1+2 合併評估（論文做法：兩階段「降噪」完成後才量化）
        print("\n--- Stage 1+2 合併去噪指標 (vs random outliers only) ---")
        calculate_denoising_metrics(len(pcd_noisy.points), noise_indices, inlier_indices_denoised)

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = os.path.join(output_dir, f"{data_name}_confounding_removed_{timestamp}.ply")
        o3d.io.write_point_cloud(filename, pcd_confounding_removed)
        print(f"  已儲存: {filename}")

        # 視覺化並截圖 (移除混合點)
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="2. Confounding Points Removed", width=1280, height=720)
        vis.add_geometry(pcd_confounding_removed)
        vis.run()
        # 截圖邏輯
        save_path = os.path.join("visualizations", "移除混合點", f"{data_name}_confounding_removed_{datetime.now().strftime('%Y%m%d-%H%M%S')}.png")
        if not os.path.exists(os.path.dirname(save_path)): os.makedirs(os.path.dirname(save_path))
        vis.capture_screen_image(save_path)
        print(f"  [視覺化截圖] 已儲存至: {save_path}")
        vis.destroy_window()

        # --- 步驟 3.3: 填補孔洞 (2D Delaunay 投影法) ---
        print("3. Stage 3: 2D Delaunay 投影法填補孔洞...")
        pcd_holes_filled = fill_holes_delaunay2d(pcd_confounding_removed,
            area_threshold_factor=3.0, max_iterations=200)
        pcd_holes_filled.paint_uniform_color([1, 1, 0]) # 黃色
        print(f"  填補孔洞後的點數: {len(pcd_holes_filled.points)}")

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = os.path.join(output_dir, f"{data_name}_holes_filled_{timestamp}.ply")
        o3d.io.write_point_cloud(filename, pcd_holes_filled)
        print(f"  已儲存: {filename}")

        # 視覺化並截圖 (填補孔洞 - 2D Delaunay)
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="3. Holes Filled (2D Delaunay)", width=1280, height=720)
        vis.add_geometry(pcd_holes_filled)
        vis.run()
        save_path = os.path.join("visualizations", "填補孔洞",
            f"{data_name}_delaunay2d_holes_filled_{datetime.now().strftime('%Y%m%d-%H%M%S')}.png")
        if not os.path.exists(os.path.dirname(save_path)): os.makedirs(os.path.dirname(save_path))
        vis.capture_screen_image(save_path)
        print(f"  [視覺化截圖] 已儲存至: {save_path}")
        vis.destroy_window()

        # --- 步驟 3.4: 手動確認並執行後處理清潔 ---
        print("\n--- 手動確認 ---")
        user_input = input("孔洞填補已完成。是否要執行最後的清潔步驟 (移除噪點與平滑化)？ (Y/n): ").lower().strip()
        
        pcd_final = pcd_holes_filled 

        if user_input != 'n':
            print("\n4. 正在對孔洞填補結果進行最後的清潔...")
            pcd_cleaned, _ = remove_outliers(pcd_holes_filled, k=20, threshold_multiplier=1.0,
                                                      density_mode="octree", threshold_method="mad")
            
            pcd_final_smoothed = pcd_cleaned
            smoothing_iterations = 3
            smoothing_k = 30
            for i in range(smoothing_iterations):
                print(f"  平滑化迭代 {i+1}/{smoothing_iterations}...")
                pcd_final_smoothed = remove_confounding_points(pcd_final_smoothed, k=smoothing_k)
            
            pcd_final = pcd_final_smoothed
            pcd_final.paint_uniform_color([0, 1, 1]) # 清潔後使用青色
            print(f"  清潔後的點數: {len(pcd_final.points)}")
        else:
            print("已跳過最後清潔步驟。")
            pcd_final.paint_uniform_color([1, 1, 0]) # 維持孔洞填補後的黃色

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = os.path.join(output_dir, f"{data_name}_final_{timestamp}.ply")
        o3d.io.write_point_cloud(filename, pcd_final)
        print(f"  已儲存最終結果: {filename}")

        print("\n顯示最終結果...")
        o3d.visualization.draw_geometries([pcd_final], window_name="4. Final Result")

        # # --- 5. 最終視覺化比較 ---
        # print(f"\n顯示 {noise_level}% 噪點等級的最終比較:")
        # pcd_clean_copy.paint_uniform_color([0.5, 0.5, 0.5]) # 原始: 灰色
        # pcd_noisy.paint_uniform_color([1, 0, 0]) # 帶瑕疵: 紅色
        # pcd_final.paint_uniform_color([0, 1, 0]) # 最終結果: 綠色
        
        # # 為了方便並排比較，將點雲在 x 軸上平移
        # pcd_noisy.translate((0.2, 0, 0))
        # pcd_final.translate((0.4, 0, 0))

        # o3d.visualization.draw_geometries([pcd_clean_copy, pcd_noisy, pcd_final],
        #                                   window_name=f"最終比較 (由左至右): 乾淨, 帶瑕疵, 處理後")

if __name__ == "__main__":
    main()
