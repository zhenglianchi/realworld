import numpy as np
import time

class Fast_PathPlanner:
    def __init__(self, planner_config):
        self.config = planner_config
        self.radius = self.config.fast_radius
        self.num_candidates = self.config.fast_num_candidates
        self.beta = self.config.fast_beta             # 方向权重
        self.alpha = self.config.fast_alpha           # 可达性权重
        self.avoid_weight = self.config.avoid_weight  # 避障惩罚权重

        # Adaptive weights (ADA method) - updated by LMP
        self.adaptive_beta = None
        self.adaptive_avoid_weight = None

    def generate_fast_point_3d_vectorized(self, current_pos, share_queue, affordable_map, avoidance_map):
        _fast_start = time.time()
        current_pos = np.array(current_pos, dtype=int)
        slow_points = np.array(share_queue.get_all().copy())
        #print(slow_points)

        # 1. 找到清除最近点后的最远的慢系统点作为引导方向
        distances = np.linalg.norm(slow_points - current_pos, axis=1)
        nearest_idx = np.argmin(distances)  # 最近点
        # 将新更新的前面的点删除
        slow_points = slow_points[nearest_idx:]
        distances = distances[nearest_idx:]

        # 找最远的点
        in_range_mask = distances <= self.radius
        candidates_in_range = slow_points[in_range_mask]
        distances_in_range = distances[in_range_mask]
        #print(f"distances: {distances}")
        if len(candidates_in_range) > 0:
            nearest_idx = np.argmax(distances_in_range)  # 最远点
            slow_target = candidates_in_range[nearest_idx].astype(int)
            #print(f"Using farthest in-range slow point: {slow_target}, distance: {distances_in_range[nearest_idx]:.2f}")
        else:
            # 回退策略：如果没有点在范围内，使用下一个点作为引导方向
            nearest_idx = 0
            slow_target = slow_points[nearest_idx].astype(int)
            #print(f"No slow point in radius {self.radius}, using the next point: {slow_target}")

        for i in range(nearest_idx+1):
            share_queue.remove_front()

        direction_vec = slow_target - current_pos
        direction_norm = np.linalg.norm(direction_vec)
        if direction_norm < 1e-6:
            print("[LOG][快规划] 目标已到达，跳过规划")
            return current_pos.copy(), slow_points.copy()
        unit_dir = direction_vec / direction_norm

        # 🔁 新增：动态半径 = 当前到目标的距离（限制在 [1, 5]）
        #dynamic_radius = max(1, min(2, int(np.ceil(direction_norm))))

        # 2. 生成候选点（在 dynamic_radius 范围内）
        r = 1
        offsets = np.random.randint(-r, r+1, size=(self.num_candidates, 3))
        candidates = current_pos + offsets  # (N, 3)

        #print(f"dynamic_radius: {dynamic_radius}")
        #print(f"candidates: {candidates}")

        # 3. 过滤非法候选点（边界 + 不等于当前点）
        shape = affordable_map.shape  # 推荐使用 shape 而非硬编码 100
        valid_mask = (
            (candidates[:, 0] >= 0) & (candidates[:, 0] < shape[0]) &
            (candidates[:, 1] >= 0) & (candidates[:, 1] < shape[1]) &
            (candidates[:, 2] >= 0) & (candidates[:, 2] < shape[2]) &
            ~((candidates == current_pos).all(axis=1))
        )
        valid_candidates = candidates[valid_mask]
        valid_candidates = valid_candidates.astype(int)
        #print(f"valid_candidates after filtering: {valid_candidates}")

        # ✅ 新增：强制加入 slow_target（如果尚未包含）
        if not np.any((valid_candidates == slow_target).all(axis=1)):
            # 检查 slow_target 是否合法（在地图范围内）
            if (0 <= slow_target[0] < shape[0] and 
                0 <= slow_target[1] < shape[1] and 
                0 <= slow_target[2] < shape[2]):
                valid_candidates = np.vstack([valid_candidates, slow_target])
                #print(f"Added slow_target {slow_target} to candidates")
            else:
                #print(f"slow_target {slow_target} is out of bounds, not added")
                pass
        else:
            #print(f"slow_target {slow_target} already in candidates")
            pass

        if len(valid_candidates) == 0:
            print("[LOG][快规划] 无有效候选点，返回当前位置")
            return current_pos.copy()

        # 4. 计算方向一致性得分
        offsets_valid = valid_candidates - current_pos
        norms = np.linalg.norm(offsets_valid, axis=1).reshape(-1, 1) + 1e-6
        unit_offsets = offsets_valid / norms
        alignment = np.sum(unit_offsets * unit_dir, axis=1)  # [-1, 1]
        angle_penalty = 1.0 - alignment  # 越小越好
        #print(f"angle_penalty: {angle_penalty}")

        # 5. 获取 affordable 得分
        afford_values = affordable_map[
            valid_candidates[:, 0],
            valid_candidates[:, 1],
            valid_candidates[:, 2]
        ]

        # 6. 获取 avoid 得分
        avoid_values = avoidance_map[
            valid_candidates[:, 0],
            valid_candidates[:, 1],
            valid_candidates[:, 2]
        ]

        # 7. 综合评分,越小越好
        # Use adaptive weights if available (ADA method), else use fixed weights
        beta_used = self.adaptive_beta if self.adaptive_beta is not None else self.beta
        avoid_used = self.adaptive_avoid_weight if self.adaptive_avoid_weight is not None else self.avoid_weight

        total_scores = (
            beta_used * angle_penalty +
            self.alpha * afford_values +
            avoid_used * avoid_values
        )

        # 8. 选择最优
        best_idx = np.argmin(total_scores)
        best_point = valid_candidates[best_idx]

        _fast_elapsed = time.time() - _fast_start
        print(f"[LOG][快规划] 快系统局部选择延时: {_fast_elapsed*1000:.1f}ms, 候选点数: {len(valid_candidates)}")
        return best_point, slow_points.copy()