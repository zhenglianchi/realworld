import numpy as np
import matplotlib.pyplot as plt

def create_distance_field(size=200, target_point=None, normalize=True):
    """
    创建欧氏距离场（目标点为0，其余为距离），可选正则化到[0,1]。
    """
    if target_point is None:
        target_point = (size // 2, size // 2, size // 2)
    tx, ty, tz = target_point

    # 使用广播高效计算距离（不显式创建 meshgrid）
    x = np.arange(size).reshape(-1, 1, 1)
    y = np.arange(size).reshape(1, -1, 1)
    z = np.arange(size).reshape(1, 1, -1)
    
    dist = np.sqrt((x - tx)**2 + (y - ty)**2 + (z - tz)**2)
    
    if normalize:
        dist = dist / dist.max()
    return dist.astype(np.float32)

def add_cube_inplace(voxel_grid, cube_size=20, cube_center=None):
    """在体素图中添加立方体（原地修改）"""
    size = voxel_grid.shape[0]
    if cube_center is None:
        cube_center = (size // 4, size // 4, size // 4)
    cx, cy, cz = cube_center
    h = cube_size // 2
    x1, x2 = max(0, cx - h), min(size, cx + h)
    y1, y2 = max(0, cy - h), min(size, cy + h)
    z1, z2 = max(0, cz - h), min(size, cz + h)
    voxel_grid[x1:x2, y1:y2, z1:z2] = 1.0

def add_cylinder_inplace(voxel_grid, radius=8, height=24, base_center=None):
    """在体素图中添加沿Z轴的圆柱体（原地修改）"""
    size = voxel_grid.shape[0]
    if base_center is None:
        base_center = (3*size//4, 3*size//4, height//2)
    cx, cy, cz = base_center  # cz 是圆柱底面中心的 z 坐标

    # 仅在圆柱区域计算，避免全图遍历
    x_min = max(0, int(cx - radius - 1))
    x_max = min(size, int(cx + radius + 2))
    y_min = max(0, int(cy - radius - 1))
    y_max = min(size, int(cy + radius + 2))
    z_min = max(0, cz - height//2)
    z_max = min(size, cz + height//2)

    print(x_min, x_max, y_min, y_max, z_min, z_max)

    x = np.arange(x_min, x_max)
    y = np.arange(y_min, y_max)
    z = np.arange(z_min, z_max)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    mask = (X - cx)**2 + (Y - cy)**2 <= radius**2
    voxel_grid[X[mask], Y[mask], Z[mask]] = 1.0

def visualize_voxel_slice(voxel_grid, axis='z', index=None, title="Voxel Slice"):
    """可视化体素图的一个切片（2D），避免3D卡顿"""
    size = voxel_grid.shape[0]
    if index is None:
        index = size // 2
    if axis == 'x':
        slice_data = voxel_grid[index, :, :]
    elif axis == 'y':
        slice_data = voxel_grid[:, index, :]
    else:  # 'z'
        slice_data = voxel_grid[:, :, index]
    
    plt.figure(figsize=(6, 5))
    plt.imshow(slice_data.T, origin='lower', cmap='viridis')
    plt.colorbar(label='Voxel Value')
    plt.title(f"{title} (slice {axis}={index})")
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

def visualize_sparse_3d(voxel, target_point, cube_center, cube_size,
                        cylinder_base_center, cylinder_radius, cylinder_height,
                        sample_ratio=0.01):
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d.art3d import Line3D
    import itertools

    # 1. 可视化稀疏体素点（非零且非障碍物区域）
    indices = np.where((voxel >= 0) & (voxel <= 1))  # 排除障碍物（=1.0）和目标点附近极小值
    total = len(indices[0])
    if total == 0:
        print("No background points to visualize.")
    else:
        n_sample = min(int(total * sample_ratio), 3000)
        idx = np.random.choice(total, n_sample, replace=False)
        x, y, z = indices[0][idx], indices[1][idx], indices[2][idx]
        colors = voxel[x, y, z]

    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制背景距离场点
    if total > 0:
        ax.scatter(x, y, z, c=colors, cmap='viridis', s=20, alpha=0.6)

    # 2. 绘制目标点（黑色，大尺寸）
    tx, ty, tz = target_point
    ax.scatter([tx], [ty], [tz], color='black', s=200, label='Target Point', edgecolors='white', linewidth=1)

    # 3. 绘制立方体边界框
    cx, cy, cz = cube_center
    h = cube_size // 2
    x1, x2 = cx - h, cx + h
    y1, y2 = cy - h, cy + h
    z1, z2 = cz - h, cz + h
    cube_corners = np.array(list(itertools.product([x1, x2], [y1, y2], [z1, z2])))
    # 12 条边
    edges = [
        (0, 1), (0, 2), (0, 4),
        (1, 3), (1, 5),
        (2, 3), (2, 6),
        (3, 7),
        (4, 5), (4, 6),
        (5, 7),
        (6, 7)
    ]
    for edge in edges:
        p1, p2 = cube_corners[edge[0]], cube_corners[edge[1]]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color='red', linewidth=2)
    ax.text(cx, cy, cz + h + 5, 'Cube Obstacle', color='red', fontsize=10)

    # 4. 绘制圆柱体边界框（用上下底面圆 + 4 条侧棱近似）
    radius = cylinder_radius
    height = cylinder_height
    cz_base = cylinder_base_center[2]
    z_bottom = cz_base - height // 2
    z_top = cz_base + height // 2
    cx_cyl, cy_cyl = cylinder_base_center[0], cylinder_base_center[1]

    # 用 8 个点近似圆
    angles = np.linspace(0, 2 * np.pi, 9)[:-1]
    bottom_circle = np.array([[cx_cyl + radius * np.cos(a), cy_cyl + radius * np.sin(a), z_bottom] for a in angles])
    top_circle    = np.array([[cx_cyl + radius * np.cos(a), cy_cyl + radius * np.sin(a), z_top]    for a in angles])

    # 绘制上下底圆
    for i in range(len(bottom_circle)):
        ax.plot([bottom_circle[i][0], bottom_circle[(i+1)%len(bottom_circle)][0]],
                [bottom_circle[i][1], bottom_circle[(i+1)%len(bottom_circle)][1]],
                [bottom_circle[i][2], bottom_circle[i][2]], color='blue', linewidth=2)
        ax.plot([top_circle[i][0], top_circle[(i+1)%len(top_circle)][0]],
                [top_circle[i][1], top_circle[(i+1)%len(top_circle)][1]],
                [top_circle[i][2], top_circle[i][2]], color='blue', linewidth=2)
        # 侧棱
        ax.plot([bottom_circle[i][0], top_circle[i][0]],
                [bottom_circle[i][1], top_circle[i][1]],
                [bottom_circle[i][2], top_circle[i][2]], color='blue', linewidth=2)

    ax.text(cx_cyl, cy_cyl, z_top + 5, 'Cylinder Obstacle', color='blue', fontsize=10)

    # 设置标签和图例
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend(loc='upper left')
    plt.title('3D Voxel Field with Target and Obstacle Bounding Boxes')
    plt.tight_layout()
    plt.show()

def add_rotated_cube_inplace(voxel_grid, cube_size=20, cube_center=None, angle_deg=0, axis='z'):
    """
    在体素图中添加一个可绕指定轴旋转的立方体（原地修改）。
    
    Parameters:
        voxel_grid: 3D numpy array (assumed cubic, shape [S, S, S])
        cube_size: int，立方体边长（体素数），必须为偶数或奇数均可
        cube_center: (cx, cy, cz) 中心坐标
        angle_deg: 旋转角度（度），**顺时针为正**（符合你的要求）
        axis: 旋转轴，目前支持 'z'（默认），可扩展 'x' 或 'y'
    """
    size = voxel_grid.shape[0]
    if cube_center is None:
        cube_center = (size // 2, size // 2, size // 2)
    cx, cy, cz = cube_center

    # 1. 生成原始立方体内的所有整数坐标（相对于中心）
    half = cube_size / 2.0
    # 使用整数坐标：例如 cube_size=4 -> [-2,-1,0,1]（中心在 0）
    coords = np.arange(-int(np.floor((cube_size - 1) / 2)), 
                       int(np.ceil((cube_size) / 2)))  # 覆盖 cube_size 个体素
    X, Y, Z = np.meshgrid(coords, coords, coords, indexing='ij')
    points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T  # (N, 3)

    # 2. 应用旋转（目前仅实现绕 Z 轴）
    if axis.lower() == 'z':
        theta = np.radians(-angle_deg)  # 顺时针 = 负角度（数学标准）
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        R = np.array([[cos_t, -sin_t, 0],
                      [sin_t,  cos_t, 0],
                      [0,      0,     1]])
    elif axis.lower() == 'y':
        theta = np.radians(-angle_deg)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        R = np.array([[cos_t, 0, sin_t],
                      [0,     1, 0],
                      [-sin_t,0, cos_t]])
    elif axis.lower() == 'x':
        theta = np.radians(-angle_deg)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        R = np.array([[1, 0,      0],
                      [0, cos_t, -sin_t],
                      [0, sin_t, cos_t]])
    else:
        raise ValueError("Only x, y, z axes are supported.")

    rotated_points = points @ R.T  # (N, 3)

    # 3. 平移到世界坐标
    world_points = rotated_points + np.array([cx, cy, cz])

    # 4. 四舍五入到最近整数体素，并裁剪到有效范围
    voxels = np.round(world_points).astype(int)
    valid = (
        (voxels[:, 0] >= 0) & (voxels[:, 0] < size) &
        (voxels[:, 1] >= 0) & (voxels[:, 1] < size) &
        (voxels[:, 2] >= 0) & (voxels[:, 2] < size)
    )
    voxels = voxels[valid]

    # 5. 标记为障碍物
    voxel_grid[voxels[:, 0], voxels[:, 1], voxels[:, 2]] = 1.0

if __name__ == "__main__":
    SIZE = 200
    target = (124, 50, 5)

    # 创建距离场
    voxel = create_distance_field(size=SIZE, target_point=target, normalize=True)

    # 障碍物参数（显式保存以便可视化）
    cube_center = (80, 90, 15)
    cube_size = 30
    cylinder_center = (140, 95, 15)  # 注意：这是底面中心的 z 坐标，与 add_cylinder_inplace 一致
    cylinder_radius = 8
    cylinder_height = 40

    # 替换原来的：
    # add_cube_inplace(voxel, cube_size=30, cube_center=(80, 90, 15))

    # 改为（例如顺时针旋转 30 度）：
    add_rotated_cube_inplace(
        voxel,
        cube_size=cube_size,
        cube_center=cube_center,
        angle_deg=60,    # 顺时针 30 度
        axis='z'
    )
    add_cylinder_inplace(voxel, radius=cylinder_radius, height=cylinder_height, base_center=cylinder_center)

    # 可视化（取消注释）
    visualize_sparse_3d(
        voxel,
        target_point=target,
        cube_center=cube_center,
        cube_size=cube_size,
        cylinder_base_center=cylinder_center,
        cylinder_radius=cylinder_radius,
        cylinder_height=cylinder_height,
        sample_ratio=0.005
    )