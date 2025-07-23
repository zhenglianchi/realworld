"""Greedy path planner."""
import numpy as np

class Fast_PathPlanner:
    """
    A greedy path planner that greedily chooses the next voxel with the lowest cost.
    Then apply several postprocessing steps to the path.
    (TODO: can be improved using more principled methods, including extension to whole-arm planning)
    """
    def __init__(self, planner_config):
        self.config = planner_config
        self.map_size = self.config.map_size

    def generate_fast_point_3d_vectorized(self, current_pos, slow_points, cost_map):
        """
        使用向量化加速，在 current_pos 附近随机采样候选点，选择最优 fast point。
        
        Parameters:
        - current_pos: 当前三维坐标 (x, y, z)
        - slow_points: List of global (x, y, z) 的慢系统点
        - cost_map: 3D numpy array, shape (400, 400, 400)
        - alpha, beta: 权重系数
        - num_candidates: 候选点数量
        - radius: 候选点采样范围（相对于 current_pos）
        Returns:
        - fast_point: tuple(x, y, z)
        """
        alpha = self.config.fast_alpha
        beta = self.config.fast_beta
        num_candidates = self.config.fast_num_candidates
        radius = self.config.fast_radius
    
        map_shape = np.array(cost_map.shape)
        current_pos = np.array(current_pos, dtype=int)

        # 1. 找到最近的慢系统目标点
        slow_points_array = np.array(slow_points)
        distances = np.linalg.norm(slow_points_array - current_pos, axis=1)
        slow_target = slow_points_array[np.argmin(distances)]
        direction_vec = slow_target - current_pos
        direction_vec = direction_vec / (np.linalg.norm(direction_vec) + 1e-6)

        # 2. 随机采样候选点（向量化）
        np.random.seed(None)  # 可选：去掉固定种子
        offsets = np.random.randint(-radius, radius + 1, size=(num_candidates * 2, 3))
        candidates = current_pos + offsets

        # 3. 筛选合法候选点（在地图范围内，且不等于当前点）
        valid_mask = np.all((candidates >= 0) & (candidates < map_shape), axis=1)
        valid_mask &= np.any(candidates != current_pos, axis=1)
        valid_candidates = candidates[valid_mask][:num_candidates]

        # 4. 批量计算偏移向量和方向一致性得分
        offsets = valid_candidates - current_pos
        offset_norms = np.linalg.norm(offsets, axis=1).reshape(-1, 1)
        unit_offsets = offsets / (offset_norms + 1e-6)

        dot_products = np.sum(unit_offsets * direction_vec, axis=1)
        angle_scores = 1.0 - dot_products

        # 5. 批量查询代价图
        costs = cost_map[valid_candidates[:, 0], valid_candidates[:, 1], valid_candidates[:, 2]]

        # 6. 计算总得分
        total_scores = alpha * costs + beta * angle_scores

        # 7. 找到最优点
        best_index = np.argmin(total_scores)

        best_point = valid_candidates[best_index]

        return best_point