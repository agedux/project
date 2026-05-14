# pc_repair_node 優化設計文檔

> 基於論文 "Optimization Algorithm for Point Cloud Quality Enhancement Based on Statistical Filtering"
> 目標：將 pc_repair_node.py 的三階段架構更貼近論文方法論

---

## Stage 1: 離群點移除 (Outlier Removal)

### 論文方法

δ = Σ||pi - pj|| / (k × ρn)

其中 ρn 為點 pi 所在 octree grid 的密度。δ 越大代表該點越可能是離群點。

### 當前實作問題

- `octree_max_depth=10` hardcode，無法適應不同尺度的點雲
- `k=15` hardcode，無法透過 ROS param 調整
- 僅使用 Open3D 固定深度 octree，與論文「自適應八叉樹」精神有落差

### 改造方案：雙模式密度計算

提供兩種密度計算模式，透過 ROS param `density_mode` 切換。

---

#### 模式 A: 自適應 Octree（`density_mode="octree"`）

**概念**：根據點雲實際尺度自動決定 octree 深度。

**深度計算公式**：

```
diag = ||bbox_max - bbox_min||          # bounding box 對角線
avg_spacing = mean(nearest_neighbor_distance)  # 平均點距
depth = log2(diag / (avg_spacing × 3.0))
depth = clamp(depth, 3, 12)
```

**邏輯**：leaf voxel 邊長約為 `avg_spacing × 3.0`，確保密集區域的 cell 內有多個點（高 ρn），稀疏 outlier 區域的 cell 內點數少（低 ρn），維持 δ 公式的鑑別力。

**參數**：

| Param | 預設 | 說明 |
|---|---|---|
| `density_octree_depth` | `-1` | `-1` = 自動計算；正整數 = 手動固定 depth |

---

#### 模式 B: Radius 密度（`density_mode="radius"`）

**概念**：放棄 octree 網格，改用固定半徑內的鄰居數作為局部密度。

**計算方式**：

```
radius = avg_spacing × density_radius_multiplier  # 預設 ×3.0
ρn(i) = count( neighbors within radius of point i )
```

**優勢**：
- 完全自適應，不受網格邊界影響
- 密度值直接反映局部點的分佈，不需要糾結 octree 切割深度
- 對不均勻點雲更 robust

**劣勢**：
- 計算量較 octree 大（需 radius search），但可用 KDTree 加速至 O(N log N)

**參數**：

| Param | 預設 | 說明 |
|---|---|---|
| `density_radius_multiplier` | `3.0` | radius = avg_spacing × multiplier |

---

### 閾值設定：雙模式

論文對 outlier 的 δ 值特徵描述：「離群點的 δ 會相當大」。但這些極端值會嚴重汙染 `mean` 和 `std`，導致 Z-score 閾值被拉高、砍不夠。

提供兩種閾值方法，透過 `threshold_method` 切換。

---

#### 模式 Z: Z-score（`threshold_method="zscore"`）

```
delta_threshold = mean(deltas) + outlier_threshold × std(deltas)
```

**問題**：`mean` 和 `std` 被極端 outlier δ 值汙染 → 閾值過高 → 漏砍。

---

#### 模式 M: MAD（`threshold_method="mad"`）

**Median Absolute Deviation** — 專為 outlier detection 設計的穩健統計量。

```
median_delta  = median(deltas)
MAD           = median(|deltas - median_delta|)
threshold     = median_delta + outlier_threshold × MAD
```

| | Z-score (mean+std) | MAD (median+MAD) |
|---|---|---|
| 受極端 outlier 汙染 | 嚴重 | 幾乎不受影響 |
| 假設分佈 | 常態分佈 | 無假設 |
| 適用場景 | 一般用途 | **專門給 outlier detection** |

一組極端 δ = [50, 80, 120] 可以把 mean 推高 10 倍，但 median 完全不動 — 這就是 MAD 的優勢。

---

### 共用參數（兩種密度模式 + 兩種閾值模式通用）

| Param | 預設 | 說明 |
|---|---|---|
| `outlier_removal_iterations` | `0` | 離群點移除迭代次數（0 = 關閉） |
| `outlier_k` | `15` | KNN 搜尋的 k 值 |
| `outlier_threshold` | `2.5` | 閾值倍數（MAD 或 Z-score 的 N 值） |
| `density_mode` | `"octree"` | 密度計算模式：`"octree"` 或 `"radius"` |
| `threshold_method` | `"mad"` | 閾值模式：`"mad"` 或 `"zscore"` |

### 安全檢查（保持不變）

若存活點數 < 原始點數 × 10%，放棄本次過濾，返回原始點雲。

---

## Stage 2: 混淆點移除 (Confounding Point Removal)

### 論文方法

對每個點 pi：
1. 搜尋 K 個鄰近點
2. 最小平方法擬合局部平面 L（法向量 n）
3. 將 pi 投影到 L：`pi' = pi - d × n`
4. 重複直到所有點處理完

### 當前實作

使用 PCA（協方差矩陣 → 最小特徵向量 = 法向量）實現局部平面擬合。評估優於論文 `Z = Ax + By + C` 公式（無座標軸依賴性，垂直面不會炸）。

### 改造方案

保留 PCA 方法，僅暴露 K 值為 ROS param。

**參數**：

| Param | 預設 | 說明 |
|---|---|---|
| `confounding_removal_iterations` | `0` | 混淆點移除迭代次數（0 = 關閉） |
| `confounding_k` | `20` | 局部平面擬合的 K 鄰居數 |

K 值特性：
- K 小（5-10）：保留細節/邊緣，但對 noise 敏感
- K 大（30-50）：平滑效果強，但尖角可能被磨圓

---

## Stage 3: 孔洞填補 (Hole Filling)

### 論文方法

1. 點雲三角化 → 網格
2. 用 Heron 公式計算每個三角形面積 si
3. si = 優先級（面積越大 = 破孔越大）
4. 在最大三角形的重心插入新點
5. 重複直到破孔完全修復

### 為何舊版論文方法失敗

| 問題 | 根因 |
|---|---|
| 大洞補不起來 | 使用 alpha-shape 做三角化，alpha-shape 會忠實地把大洞判定為「外部空間」，不在該處生成三角形 → 演算法看不到這個洞 |
| 補洞速度極慢 | 逐點插入 + 每輪重建 alpha-shape，千次迭代數十分鐘 |
| 補點不自然 | 新點只在三角形中心插入，未貼合物體表面 |

**核心矛盾**：論文需要「整個凸包都有三角形，大三角形 = 破孔」，但 alpha-shape 專門刪除大三角形。兩者互斥。

### 改造方案：2D Delaunay 投影法

適用於 eye-to-hand 桌面場景（點雲本質為 2.5D，無 overhang）。

**流程**：

```
1. PCA 找點雲主平面（前兩個 principal component）
2. 3D 點投影到 2D 平面
3. 2D Delaunay triangulation → 凸包內全部有三角形覆蓋
4. 計算每個 2D 三角形面積
5. 面積 > threshold 的三角形 = 破孔
6. 在 2D 重心處插新點（2D 座標）
7. 將 2D 座標 + Z（3 頂點 IDW 加權）拉回 3D
8. 重複步驟 3-7 直到最大面積 < 閾值
```

**為何能解決 alpha-shape 的問題**：

- Delaunay **一定會**在凸包內填滿三角形 → 大洞一定有對應的大三角形
- 2D Delaunay 只生成三角形（非四面體）→ 避免 3D Delaunay 的內部四面體問題
- 面積過濾就是論文的「優先級」邏輯

**限制**：

- 對非 2.5D 形狀（overhang、垂直面、複雜摺疊）可能失真
- bunny/armadillo 是 3D 模型，測試時需關注這點

**參數**：

| Param | 預設 | 說明 |
|---|---|---|
| `hole_filling_iterations` | `0` | 孔洞填補迭代次數（0 = 關閉） |
| `hole_fill_mode` | `"delaunay2d"` | 填補模式：`"delaunay2d"` 或 `"bpa"`（保留舊版） |
| `hole_area_threshold_factor` | `3.0` | 面積閾值 = avg_triangle_area × factor |
| `hole_max_iterations` | `100` | 每次 fill 的最大插入點數 |
