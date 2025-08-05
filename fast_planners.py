import numpy as np

class Fast_PathPlanner:
    def __init__(self, planner_config):
        self.config = planner_config
        self.radius = self.config.fast_radius          # 推荐 2
        self.num_candidates = self.config.fast_num_candidates  # 推荐 50
        self.beta = self.config.fast_beta             # 方向权重
        self.fast_alpha = self.config.fast_alpha
        self.avoid_weight = self.config.avoid_weight  # 避障惩罚权重

    def generate_fast_point_3d_vectorized(self, current_pos, slow_points, affordable_map, avoidance_map):
        """
        快速局部决策：选择下一个执行点
        输入均为 voxel 坐标，affordable_map 值越小越好，avoidance_map=1 为禁区
        
        Parameters:
        - current_pos: (x, y, z) 当前位置 (int)
        - slow_points: List[np.ndarray] 慢系统路径点 (N, 3), voxel 坐标
        - affordable_map: (100,100,100) float, 0=目标, 1=不可去
        - avoidance_map: (100,100,100) binary, 1=禁止进入
        
        Returns:
        - next_pos: (x, y, z) 下一个 voxel 坐标
        """
        current_pos = np.array(current_pos, dtype=int)
        slow_points = np.array(slow_points)  # 确保是 numpy array

        # 1. 找到最近的慢系统点作为引导方向
        distances = np.linalg.norm(slow_points - current_pos, axis=1)
        if len(distances) == 0:
            return current_pos.copy()
        nearest_idx = np.argmin(distances)
        slow_target = slow_points[nearest_idx]
        
        # 如果已经非常接近目标，直接返回当前点（等待停止）
        if distances[nearest_idx] < 1.0:
            return current_pos.copy()

        direction_vec = slow_target - current_pos
        direction_norm = np.linalg.norm(direction_vec)
        if direction_norm < 1e-6:
            return current_pos.copy()
        unit_dir = direction_vec / direction_norm

        # 2. 生成候选点（在 radius 范围内）
        r = self.radius
        offsets = np.random.randint(-r, r + 1, size=(self.num_candidates, 3))
        candidates = current_pos + offsets  # (N, 3)

        # 3. 过滤非法候选点（边界 + 不等于当前点）
        valid_mask = (
            (candidates >= 0).all(axis=1) &
            (candidates < 100).all(axis=1) &
            ~((candidates == current_pos).all(axis=1))
        )
        valid_candidates = candidates[valid_mask]
        if len(valid_candidates) == 0:
            return current_pos.copy()

        # 4. 计算方向一致性得分（越接近目标方向越好）
        offsets_valid = valid_candidates - current_pos
        norms = np.linalg.norm(offsets_valid, axis=1).reshape(-1, 1) + 1e-6
        unit_offsets = offsets_valid / norms
        alignment = np.sum(unit_offsets * unit_dir, axis=1)  # [-1, 1]
        angle_penalty = 1.0 - alignment  # 越小越好

        # 5. 获取 affordable 得分（✅ 值越小越好 → 直接作为成本）
        afford_values = affordable_map[
            valid_candidates[:, 0],
            valid_candidates[:, 1],
            valid_candidates[:, 2]
        ]  # shape: (M,)

        # 6. 获取 avoid 得分（✅ 1=禁止）
        avoid_values = avoidance_map[
            valid_candidates[:, 0],
            valid_candidates[:, 1],
            valid_candidates[:, 2]
        ]  # 0 或 1

        # 7. 综合评分（越小越好）
        total_scores = (
            self.beta * angle_penalty      # 方向对齐
            + self.fast_alpha * afford_values          # afford 值越小越好 → 直接加
            + self.avoid_weight * avoid_values  # avoid=1 时惩罚
        )

        # ✅ 强制避障：对 avoid=1 的点施加巨大惩罚
        danger_mask = avoid_values > 0.5
        total_scores[danger_mask] += 100.0  # 确保绝对不选

        # 8. 选择最优
        best_idx = np.argmin(total_scores)
        best_point = valid_candidates[best_idx]

        return best_point