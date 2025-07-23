import numpy as np
from fast_planners import Fast_PathPlanner
from arguments import get_config
#load config
config_path = "configs/vlm_rlbench_config.yaml"
config = get_config(config_path=config_path)

planner = Fast_PathPlanner(config["planner"])

# 创建一个 400x400x400 的代价图
cost_map = np.random.rand(400, 400, 400) * 10

# 当前位置和慢系统点
current_pos = np.array((200, 200, 200))
slow_points = np.array([(220, 220, 220), (250, 250, 250), (300, 300, 300),(300, 300, 300),(300, 300, 300)])
print(cost_map.shape)
print(current_pos.shape)
print(slow_points.shape)

print(cost_map)
print(current_pos)
print(slow_points)
# 生成快点
fast_point = planner.generate_fast_point_3d_vectorized(current_pos, slow_points, cost_map)
print("Selected Fast Point:", fast_point)