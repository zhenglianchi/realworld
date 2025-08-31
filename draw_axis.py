import cv2
import numpy as np
import matplotlib.pyplot as plt
from camera import Camera
from UR_Base import UR_BASE
import time

def draw_axis_on_image(image, K, R, t, axis_length=0.1, min_z=0.1, label="Base"):
    """
    在图像上绘制简洁的3D坐标轴（X: 红, Y: 绿, Z: 蓝）
    并在旁边标注坐标系名称（英文）
    
    :param image: 输入图像 (H, W, 3)，BGR格式
    :param K: 相机内参矩阵 (3, 3)
    :param R: 旋转矩阵 (3,3)，目标坐标系 → 相机坐标系
    :param t: 平移向量 (3,1)，目标原点在相机坐标系中的位置
    :param axis_length: 坐标轴长度（米）
    :param min_z: 最小有效深度
    :param label: 坐标系名称 ("Base" / "End-effector")
    :return: 绘制后的图像
    """
    import cv2, numpy as np

    def draw_arrow(img, p1, p2, color, thickness=2, head_size=6):
        """绘制细箭头"""
        cv2.arrowedLine(img, p1, p2, color, thickness, tipLength=0.2)

    # ---------------- 生成统一长度的坐标轴点 ----------------
    axis_points = np.array([
        [0, 0, 0],               # 原点
        [axis_length, 0, 0],     # X
        [0, axis_length, 0],     # Y
        [0, 0, axis_length],     # Z
    ])
    axis_points_cam = (R @ axis_points.T + t).T

    # 投影
    points_2d = []
    for pt in axis_points_cam:
        x, y, z = pt
        if z < min_z:
            points_2d.append(None)
            continue
        u = int((K[0, 0] * x / z + K[0, 2]))
        v = int((K[1, 1] * y / z + K[1, 2]))
        points_2d.append((u, v, z))

    if points_2d[0] is None:
        return image

    origin_2d = (points_2d[0][0], points_2d[0][1])
    frame_h, frame_w = image.shape[:2]
    if not (0 <= origin_2d[0] < frame_w and 0 <= origin_2d[1] < frame_h):
        return image

    # ---------------- 颜色 (BGR格式) ----------------
    colors = {'Z': (0, 0, 255), 'Y': (0, 255, 0), 'X': (255, 0, 0)}
    labels = ['X', 'Y', 'Z']
    endpoints = [points_2d[1], points_2d[2], points_2d[3]]

    for end_pt, (axis_name, color) in zip(endpoints, colors.items()):
        if end_pt is None:
            continue
        u, v, _ = end_pt
        end_2d = (u, v)
        draw_arrow(image, origin_2d, end_2d, color, thickness=2, head_size=8)

        # 小标签 (X, Y, Z)
        offset = (8, -8)
        cv2.putText(image, axis_name, (end_2d[0] + offset[0], end_2d[1] + offset[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255),
                    1, lineType=cv2.LINE_AA)

    # ---------------- 坐标系整体标签 ----------------
    sys_label_pos = (origin_2d[0] + 15, origin_2d[1] + 15)
    cv2.putText(image, label, sys_label_pos,
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0),
                2, lineType=cv2.LINE_AA)  # 黑底
    cv2.putText(image, label, sys_label_pos,
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255),
                1, lineType=cv2.LINE_AA)  # 白字

    return image



def draw_axis(camera, ur5, frame_rgb):

    K = camera.get_intrinsic_matrix()
    T_base_to_camera = camera.get_extrinsic_matrix()
    T_camera_to_base = np.linalg.inv(T_base_to_camera)

    frame_draw = frame_rgb.copy()

    # --- 获取当前末端位姿（TCP 相对于 Base）---
    tcp_pose = ur5.get_tcp()  # 假设返回 [x,y,z,rx,ry,rz] 或齐次矩阵
    x, y, z, rx, ry, rz = tcp_pose
    R_tcp, _ = cv2.Rodrigues(np.array([rx, ry, rz]))
    t_tcp = np.array([[x, y, z]]).T
    T_base_to_tcp = np.eye(4)
    T_base_to_tcp[:3, :3] = R_tcp
    T_base_to_tcp[:3, 3] = t_tcp.flatten()

    # --- 计算在相机坐标系下的坐标系位姿 ---

    # 1. TCP 在相机坐标系下
    T_camera_to_tcp = T_camera_to_base @ T_base_to_tcp
    R_cam_tcp = T_camera_to_tcp[:3, :3]
    t_cam_tcp = T_camera_to_tcp[:3, 3:4]

    # 2. Base 在相机坐标系下（其实就是 T_camera_to_base）
    R_cam_base = T_camera_to_base[:3, :3]
    t_cam_base = T_camera_to_base[:3, 3:4]

    # --- 绘制两个坐标系 ---
    # 绘制末端（TCP）坐标系
    frame_draw = draw_axis_on_image(
        frame_draw, K, R_cam_tcp, t_cam_tcp,
        axis_length=0.15, min_z=0.1, label="End-effector"
    )

    # 绘制基座（Base）坐标系
    frame_draw = draw_axis_on_image(
        frame_draw, K, R_cam_base, t_cam_base,
        axis_length=0.15, min_z=0.1, label="Base"
    )

    return frame_draw




