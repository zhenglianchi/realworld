"""Greedy path planner."""
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.ndimage import distance_transform_edt
from scipy.signal import savgol_filter
from utils import get_clock_time, normalize_map, calc_curvature

def generate_fast_point_3d_vectorized(current_pos, slow_points, cost_map, alpha=1.0, beta=2.0, num_candidates=20, radius=10):
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

    # 如果合法点不足，补充到 num_candidates
    while len(valid_candidates) < num_candidates:
        new_candidate = current_pos + np.random.randint(-radius, radius + 1, size=3)
        if np.all((new_candidate >= 0) & (new_candidate < map_shape)) and not np.array_equal(new_candidate, current_pos):
            valid_candidates = np.vstack([valid_candidates, new_candidate])

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

    return tuple(best_point)


class PathPlanner:
    """
    A greedy path planner that greedily chooses the next voxel with the lowest cost.
    Then apply several postprocessing steps to the path.
    (TODO: can be improved using more principled methods, including extension to whole-arm planning)
    """
    def __init__(self, planner_config, map_size):
        self.config = planner_config
        self.map_size = map_size

    def optimize(self, start_pos: np.ndarray, target_map: np.ndarray, obstacle_map: np.ndarray):
        """
        config:
            start_pos: (3,) np.ndarray, start position
            target_map: (map_size, map_size, map_size) np.ndarray, target_map
            obstacle_map: (map_size, map_size, map_size) np.ndarray, obstacle_map
        Returns:
            path: (n, 3) np.ndarray, path
        """
        # make copies
        start_pos, raw_start_pos = start_pos.copy(), start_pos
        target_map, raw_target_map = target_map.copy(), target_map
        obstacle_map, raw_obstacle_map = obstacle_map.copy(), obstacle_map
        # smoothing
        target_map = distance_transform_edt(1 - target_map)
        target_map = normalize_map(target_map)
        obstacle_map = gaussian_filter(obstacle_map, sigma=self.config.obstacle_map_gaussian_sigma)
        obstacle_map = normalize_map(obstacle_map)
        # combine target_map and obstacle_map
        costmap = target_map * self.config.target_map_weight + obstacle_map * self.config.obstacle_map_weight
        costmap = normalize_map(costmap)
        _costmap = costmap.copy()
        # get stop criteria
        stop_criteria = self._get_stop_criteria()
        # initialize path
        path, current_pos = [start_pos], start_pos
        for i in range(self.config.max_steps):
            # calculate all nearby voxels around current position
            all_nearby_voxels = self._calculate_nearby_voxel(current_pos)
            # calculate the score of all nearby voxels
            nearby_score = _costmap[all_nearby_voxels[:, 0], all_nearby_voxels[:, 1], all_nearby_voxels[:, 2]]
            # Find the minimum cost voxel
            steepest_idx = np.argmin(nearby_score)
            next_pos = all_nearby_voxels[steepest_idx]
            # increase cost at current position to avoid going back
            _costmap[current_pos[0].round().astype(int),
                     current_pos[1].round().astype(int),
                     current_pos[2].round().astype(int)] += 1
            # update path and current position
            path.append(next_pos)
            current_pos = next_pos
            # check stop criteria
            if stop_criteria(current_pos, _costmap, self.config.stop_threshold):
                break
        raw_path = np.array(path)
        # postprocess path
        processed_path = self._postprocess_path(raw_path, raw_target_map)
        return processed_path
    
    def _get_stop_criteria(self):
        def no_nearby_equal_criteria(current_pos, costmap, stop_threshold):
            """
            Do not stop if there is a nearby voxel with cost less than current cost + stop_threshold.
            """
            assert np.isnan(costmap).sum() == 0, 'costmap contains nan'
            current_pos_discrete = current_pos.round().clip(0, self.map_size - 1).astype(int)
            current_cost = costmap[current_pos_discrete[0], current_pos_discrete[1], current_pos_discrete[2]]
            nearby_locs = self._calculate_nearby_voxel(current_pos)
            nearby_equal = np.any(costmap[nearby_locs[:, 0], nearby_locs[:, 1], nearby_locs[:, 2]] < current_cost + stop_threshold)
            if nearby_equal:
                return False
            return True
        return no_nearby_equal_criteria

    def _calculate_nearby_voxel(self, current_pos):
        # create a grid of nearby voxels
        half_size = int(2 * self.map_size / 100)
        offsets = np.arange(-half_size, half_size + 1)
        # our heuristics-based dynamics model only supports planar pushing -> only xy path is considered
        offsets_grid = np.array(np.meshgrid(offsets, offsets, offsets)).T.reshape(-1, 3)
        # Remove the [0, 0, 0] offset, which corresponds to the current position
        offsets_grid = offsets_grid[np.any(offsets_grid != [0, 0, 0], axis=1)]
        # Calculate all nearby voxel coordinates
        all_nearby_voxels = np.clip(current_pos + offsets_grid, 0, self.map_size - 1)
        # Remove duplicates, if any, caused by clipping
        all_nearby_voxels = np.unique(all_nearby_voxels, axis=0)
        return all_nearby_voxels
    
    def _postprocess_path(self, path, raw_target_map):
        """
        Apply various postprocessing steps to the path.
        """
        # smooth the path
        savgol_window_size = min(len(path), self.config.savgol_window_size)
        savgol_polyorder = min(self.config.savgol_polyorder, savgol_window_size - 1)
        path = savgol_filter(path, savgol_window_size, savgol_polyorder, axis=0)
        # early cutoff if curvature is too high
        curvature = calc_curvature(path)
        if len(curvature) > 5:
            high_curvature_idx = np.where(curvature[5:] > self.config.max_curvature)[0]
            if len(high_curvature_idx) > 0:
                high_curvature_idx += 5
                path = path[:int(0.9 * high_curvature_idx[0])]  
        # skip waypoints such that they reach target spacing
        path_trimmed = path[1:-1]
        skip_ratio = None
        if len(path_trimmed) > 1:
            target_spacing = int(self.config['target_spacing'] * self.map_size / 100)
            length = np.linalg.norm(path_trimmed[1:] - path_trimmed[:-1], axis=1).sum()
            if length > target_spacing:
                curr_spacing = np.linalg.norm(path_trimmed[1:] - path_trimmed[:-1], axis=1).mean()
                skip_ratio = np.round(target_spacing / curr_spacing).astype(int)
                if skip_ratio > 1:
                    path_trimmed = path_trimmed[::skip_ratio]
        path = np.concatenate([path[0:1], path_trimmed, path[-1:]])
        # force last position to be one of the target positions
        last_waypoint = path[-1].round().clip(0, self.map_size - 1).astype(int)
        if raw_target_map[last_waypoint[0], last_waypoint[1], last_waypoint[2]] == 0:
            # find the closest target position
            target_pos = np.argwhere(raw_target_map == 1)
            closest_target_idx = np.argmin(np.linalg.norm(target_pos - last_waypoint, axis=1))
            closest_target = target_pos[closest_target_idx]
            path = np.append(path, [closest_target], axis=0)
        path = path.clip(0, self.map_size-1)
        return path
