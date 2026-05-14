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

    # Stage 1 只該砍隨機離群點（Gaussian 擾動點留給 Stage 2 修正）
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
        points_to_keep_mask = np.ones(len(points_for_holes), dtype=bool)
        points_to_keep_mask[unique_indices_to_remove] = False
        kept_old_indices = np.where(points_to_keep_mask)[0]

        # select_by_index 會重排 → ground truth 索引要 remap
        noisy_pcd = noisy_pcd.select_by_index(kept_old_indices)

        # 舊索引 → 新索引 mapping
        old_to_new = {old: new for new, old in enumerate(kept_old_indices)}
        remapped = set()
        for idx in ground_truth_noise_indices:
            if idx in old_to_new:
                remapped.add(old_to_new[idx])
        ground_truth_noise_indices = np.array(sorted(remapped))

    return noisy_pcd, ground_truth_noise_indices

def calculate_denoising_metrics(total_points_in_noisy_pcd, ground_truth_noise_indices, denoised_inlier_indices):
    """
    計算並印出去噪演算法的性能指標 (Precision, Recall, F1-score)。

    Args:
        total_points_in_noisy_pcd (int): 帶噪點雲的總點數。
        ground_truth_noise_indices (set): 真實噪點的索引集合。
        denoised_inlier_indices (set): 經過演算法處理後，被判定為「內點」(Inlier) 的索引集合。
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

def _get_octree_density_map(pcd, octree_max_depth):
    """
    [輔助函式] 使用八叉樹 (Octree) 結構計算點雲中每個點的局部密度。
    密度被定義為該點所在的最小八叉樹立方體內的點的數量。
    """
    # 建立八叉樹並從點雲進行轉換
    octree = o3d.geometry.Octree(max_depth=octree_max_depth)
    octree.convert_from_point_cloud(pcd, size_expand=0.01)

    points = np.asarray(pcd.points)
    # 建立一個字典，用於儲存每個八叉樹葉節點中的點的索引
    leaf_node_key_to_point_indices = {}

    # 遍歷所有點，將它們分配到對應的葉節點中
    for i, point in enumerate(points):
        # 找到點所在的葉節點
        leaf_node, info = octree.locate_leaf_node(point)
        if leaf_node:
            # 使用葉節點的 (原點, 大小) 作為獨一無無的鍵
            node_key = (info.origin[0], info.origin[1], info.origin[2], info.size)
            if node_key not in leaf_node_key_to_point_indices:
                leaf_node_key_to_point_indices[node_key] = []
            leaf_node_key_to_point_indices[node_key].append(i)

    # 根據每個葉節點的點數計算密度，並賦值給對應的點
    point_densities = np.zeros(len(points))
    for node_key, indices in leaf_node_key_to_point_indices.items():
        density = len(indices)
        for idx in indices:
            point_densities[idx] = density
    
    # 處理可能未被分配到任何葉節點的點，給予預設密度 1，以避免除以零的錯誤
    point_densities[point_densities == 0] = 1

    return point_densities

def remove_outliers(pcd, octree_max_depth=8, k=10, threshold_multiplier=1.0):
    """
    移除離群點。結合「八叉樹密度」與「統計分析」的演算法。
    """
    if len(pcd.points) < k + 1:
        return pcd, np.arange(len(pcd.points))
    
    points = np.asarray(pcd.points)
    
    # 步驟 1: 取得每個點的八叉樹局部密度 (pn)
    point_densities = _get_octree_density_map(pcd, octree_max_depth)

    # 步驟 2: 計算每個點與其 k 個鄰居的平均距離
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    deltas = np.zeros(len(points))
    for i in range(len(points)):
        # 搜尋 k+1 個鄰居 (包含點本身)
        [_, idx, _] = pcd_tree.search_knn_vector_3d(points[i], k + 1)
        neighbors = points[idx[1:]] # 排除點本身
        avg_distance = np.mean(np.linalg.norm(points[i] - neighbors, axis=1))
        
        density = point_densities[i]
        
        # 步驟 3: 計算離群機率 delta
        # delta = 平均鄰居距離 / 局部密度
        if density > 0:
            deltas[i] = avg_distance / density
        else:
            # 如果密度為0，視為無限大的離群機率
            deltas[i] = float('inf')
            
    if len(deltas) == 0: return pcd, np.arange(len(pcd.points))

    # 步驟 4: 建立閾值並移除離群點
    # 使用百分位數來設定一個動態閾值，使其對不同點雲更具適應性
    delta_threshold = np.percentile(deltas, 90) * threshold_multiplier
    # delta 值小於閾值的點被視為內點 (inlier)
    inlier_indices = np.where(deltas < delta_threshold)[0]
    
    return pcd.select_by_index(inlier_indices), inlier_indices

def remove_confounding_points(pcd, k=20):
    """
    移除混淆點 (平滑化)。「局部平面擬合與投影」。
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

def fill_holes_bpa(pcd):
    """
    [最終方法] 使用滾球算法 (Ball-Pivoting Algorithm, BPA) 填補孔洞。
    BPA 對於處理帶有精細細節和孔洞的點雲通常比 Alpha-shape 和泊松重建更穩健。
    """
    print("  正在使用滾球算法 (BPA) 進行網格重建...")
    
    # 強制重新計算並統一法線方向，這是解決光照問題的關鍵
    pcd.estimate_normals()
    camera_location = pcd.get_max_bound() + np.array([0, 0, np.linalg.norm(pcd.get_max_bound() - pcd.get_min_bound())])
    pcd.orient_normals_towards_camera_location(camera_location)

    # 計算滾動球體的半徑。這一步是關鍵。
    # 我們提供一系列半徑，從較小的值（捕捉細節）到較大的值（跨越孔洞）。
    avg_dist = np.mean(pcd.compute_nearest_neighbor_distance())
    radii = [avg_dist, avg_dist * 4, avg_dist * 8, avg_dist * 12, avg_dist * 16, avg_dist * 20]
    
    try:
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector(radii))
        
        # 對網格進行一些基礎的清潔
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_unreferenced_vertices()
        
        print("  BPA 網格重建完成。正在從網格採樣...")
        # 從重建的網格上採樣，以獲得均勻的點雲
        pcd_filled = mesh.sample_points_uniformly(number_of_points=len(pcd.points))
        return pcd_filled

    except Exception as e:
        print(f"  BPA 執行失敗: {e}")
        return pcd

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
        
        # --- 步驟 3.1: 移除離群點 ---
        print("1. 正在移除離群點...")
        pcd_outliers_removed, inlier_indices_denoised = remove_outliers(pcd_noisy, octree_max_depth=10, k=15, threshold_multiplier=0.85)
        
        calculate_denoising_metrics(len(pcd_noisy.points), noise_indices, inlier_indices_denoised)
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

        # --- 步驟 3.3: 填補孔洞 (使用BPA) ---
        print("3. 正在填補孔洞 (使用滾球算法 BPA)...")
        pcd_holes_filled = fill_holes_bpa(pcd_confounding_removed)
        pcd_holes_filled.paint_uniform_color([1, 1, 0]) # 黃色
        print(f"  填補孔洞後的點數: {len(pcd_holes_filled.points)}")

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = os.path.join(output_dir, f"{data_name}_holes_filled_{timestamp}.ply")
        o3d.io.write_point_cloud(filename, pcd_holes_filled)
        print(f"  已儲存: {filename}")

        # 視覺化並截圖 (填補孔洞 - BPA)
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="3. Holes Filled (BPA)", width=1280, height=720)
        vis.add_geometry(pcd_holes_filled)
        vis.run()
        # 截圖邏輯
        save_path = os.path.join("visualizations", "填補孔洞", f"{data_name}_holes_filled_{datetime.now().strftime('%Y%m%d-%H%M%S')}.png")
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
            pcd_cleaned, _ = remove_outliers(pcd_holes_filled, octree_max_depth=8, k=20, threshold_multiplier=1.0)
            
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
