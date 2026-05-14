import open3d as o3d
import numpy as np
import os
from datetime import datetime
from scipy.spatial import cKDTree


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

def remove_outliers(pcd, octree_max_depth=8, k=10, threshold_multiplier=2.0):
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
    radii = [avg_dist, avg_dist * 2, avg_dist * 4, avg_dist * 8]
    
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
    file_path = "rawdata/deleted_pc.ply" 
    
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
            
            print(f"成功載入 {len(pcd_clean.points)} 個點。 ")
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
    
    # --- 3. 執行點雲優化流程 ---
    
    # --- 步驟 3.1: 移除離群點 ---
    print("1. 正在移除離群點...")
    pcd_outliers_removed, inlier_indices_denoised = remove_outliers(pcd_clean, octree_max_depth=10, k=15, threshold_multiplier=0.5)
    
    pcd_outliers_removed.paint_uniform_color([0, 1, 0]) # 綠色
    
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = os.path.join(output_dir, f"{data_name}_outliers_removed_{timestamp}.ply")
    o3d.io.write_point_cloud(filename, pcd_outliers_removed)
    print(f"  已儲存: {filename}")

    # --- 步驟 3.2: 移除混淆點 (表面平滑化) ---
    print("2. 正在移除混淆點 (局部平面投影)...")
    pcd_smoothed = pcd_outliers_removed
    smoothing_iterations = 3 
    smoothing_k = 30 
    for i in range(smoothing_iterations):
        print(f"  平滑化迭代 {i+1}/{smoothing_iterations}...")
        pcd_smoothed = remove_confounding_points(pcd_smoothed, k=smoothing_k)

    pcd_smoothed.paint_uniform_color([0, 0, 1]) # 藍色
    print(f"  所有平滑步驟後的點數: {len(pcd_smoothed.points)}")

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = os.path.join(output_dir, f"{data_name}_confounding_removed_{timestamp}.ply")
    o3d.io.write_point_cloud(filename, pcd_smoothed)
    print(f"  已儲存: {filename}")

    o3d.visualization.draw_geometries([pcd_smoothed], window_name="2. Confounding Points Removed")

    # --- 步驟 3.3: 填補孔洞 (使用BPA) ---
    print("3. 正在填補孔洞 (使用滾球算法 BPA)...")
    pcd_holes_filled = fill_holes_bpa(pcd_smoothed)
    pcd_holes_filled.paint_uniform_color([1, 1, 0]) # 黃色
    print(f"  填補孔洞後的點數: {len(pcd_holes_filled.points)}")

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = os.path.join(output_dir, f"{data_name}_holes_filled_{timestamp}.ply")
    o3d.io.write_point_cloud(filename, pcd_holes_filled)
    print(f"  已儲存: {filename}")

    o3d.visualization.draw_geometries([pcd_holes_filled], window_name="3. Holes Filled (BPA)")

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
