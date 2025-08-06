# voxel_scene_visualizer.py

import os
import numpy as np
import plotly.graph_objects as go
from plotly.offline import plot

class VoxelSceneVisualizer:
    def __init__(self, save_dir="./visualizations"):
        """
        :param save_dir: HTML 文件保存目录
        """
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def visualize(
        self,
        scenemap,
        costmap,
        executed_path_voxel,
        filename="scene_path.html",
        max_scene_points=500,
        max_costmap_points=500,
        show_cost_text=True
    ):
        """
        在体素坐标系下可视化场景图、成本图和快系统路径。

        :param scenemap: (D, H, W) bool or binary array，表示哪些体素是“场景”
        :param costmap: (D, H, W) float array，表示每个体素的 cost
        :param executed_path_voxel: list of [x, y, z]，快系统路径（体素坐标）
        :param filename: 输出 HTML 文件名
        :param max_scene_points: 最多显示的场景点数（降采样）
        :param max_costmap_points: 最多显示的成本点数（降采样）
        :param show_cost_text: 是否悬停显示 cost
        """
        fig = go.Figure()

        # === 1. 可视化 scenemap（环境结构）
        scene_indices = np.array(np.where(scenemap > 0)).T
        if len(scene_indices) == 0:
            print("[Visualizer] Warning: scenemap is empty.")
        else:
            if len(scene_indices) > max_scene_points:
                scene_indices = scene_indices[np.random.choice(len(scene_indices), max_scene_points, replace=False)]

            fig.add_trace(go.Scatter3d(
                x=scene_indices[:, 0], y=scene_indices[:, 1], z=scene_indices[:, 2],
                mode='markers',
                marker=dict(size=2, color='gray', opacity=0.5),
                name='Scene Structure'
            ))

        # === 2. 可视化 costmap（非零区域）
        cost_indices = np.array(np.where(costmap > 0)).T
        if len(cost_indices) == 0:
            print("[Visualizer] Warning: costmap is all zero.")
        else:
            if len(cost_indices) > max_costmap_points:
                cost_indices = cost_indices[np.random.choice(len(cost_indices), max_costmap_points, replace=False)]

            costs = costmap[cost_indices[:, 0], cost_indices[:, 1], cost_indices[:, 2]]

            hover_text = [
                f"Voxel: ({i[0]}, {i[1]}, {i[2]})<br>Cost: {c:.4f}"
                for i, c in zip(cost_indices, costs)
            ] if show_cost_text else None

            fig.add_trace(go.Scatter3d(
                x=cost_indices[:, 0], y=cost_indices[:, 1], z=cost_indices[:, 2],
                mode='markers',
                marker=dict(
                    size=3,
                    color=costs,
                    colorscale='Viridis',
                    colorbar=dict(title="Cost"),
                    opacity=0.8
                ),
                text=hover_text,
                hoverinfo="text",
                name='Cost Map'
            ))

        # === 3. 可视化快系统路径
        if not executed_path_voxel or len(executed_path_voxel) == 0:
            print("[Visualizer] Warning: executed_path_voxel is empty.")
        else:
            path = np.array(executed_path_voxel)
            valid_mask = np.all((path >= 0) & (path < np.array(costmap.shape)), axis=1)
            path = path[valid_mask]

            if len(path) == 0:
                print("[Visualizer] No valid path points in map range.")
            else:
                path_costs = [costmap[p[0], p[1], p[2]] for p in path]
                path_hover = [
                    f"Step {i}<br>Voxel: ({p[0]}, {p[1]}, {p[2]})<br>Cost: {c:.4f}"
                    for i, (p, c) in enumerate(zip(path, path_costs))
                ]

                fig.add_trace(go.Scatter3d(
                    x=path[:, 0], y=path[:, 1], z=path[:, 2],
                    mode='lines+markers',
                    line=dict(color='red', width=10),
                    marker=dict(size=8, color=path_costs, colorscale='Hot', colorbar=dict(title="Path Cost")),
                    text=path_hover,
                    hoverinfo="text",
                    name='Executed Path (Fast)'
                ))

        # === 布局
        fig.update_layout(
            title="3D Scene & Cost Map with Fast Path",
            scene=dict(
                xaxis_title='Voxel X',
                yaxis_title='Voxel Y',
                zaxis_title='Voxel Z',
                aspectmode='data',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            margin=dict(l=0, r=0, b=0, t=50),
            showlegend=True
        )

        # 保存
        full_path = os.path.join(self.save_dir, filename)
        plot(fig, filename=full_path, auto_open=False)
        print(f"[Visualizer] Saved to: {full_path}")