import numpy as np
from slow_planners import generate_fast_point_3d_vectorized

# 创建一个 400x400x400 的代价图
cost_map = np.random.rand(400, 400, 400) * 10

# 当前位置和慢系统点
current_pos = (200, 200, 200)
slow_points = [(220, 220, 220), (250, 250, 250), (300, 300, 300)]

# 生成快点
fast_point = generate_fast_point_3d_vectorized(current_pos, slow_points, cost_map)
print("Selected Fast Point:", fast_point)