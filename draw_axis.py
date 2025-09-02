import cv2
import numpy as np

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

    def draw_arrow(img, p1, p2, color, thickness=2):
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
    endpoints = [points_2d[1], points_2d[2], points_2d[3]]

    for end_pt, (axis_name, color) in zip(endpoints, colors.items()):
        if end_pt is None:
            continue
        u, v, _ = end_pt
        end_2d = (u, v)
        draw_arrow(image, origin_2d, end_2d, color, thickness=2, head_size=8)

        # 小标签 (X, Y, Z)
        offset = (5, -5)
        cv2.putText(image, axis_name, (end_2d[0] + offset[0], end_2d[1] + offset[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255),
                    1, lineType=cv2.LINE_AA)

    # ---------------- 坐标系整体标签 ----------------
    sys_label_pos = (origin_2d[0] + 10, origin_2d[1] + 10)
    cv2.putText(image, label, sys_label_pos,
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0),
                2, lineType=cv2.LINE_AA)  # 黑底
    cv2.putText(image, label, sys_label_pos,
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255),
                1, lineType=cv2.LINE_AA)  # 白字

    return image


def draw_fixed_base_axis(image, K, R_cam_base, image_position=None,
                         axis_length=0.05, label_scale=0.4):
    """
    在图像固定位置绘制 Base 坐标系（如左下角），用于参考
    :param image: 输入图像
    :param K: 相机内参 (3,3)
    :param R_cam_base: 旋转矩阵，Base 坐标系 → 相机坐标系
    :param image_position: 固定绘制原点 (u, v)，默认为左下角
    :param axis_length: 坐标轴长度（米），较小
    :param label_scale: 字体缩放
    :return: 绘制后的图像
    """
    h, w = image.shape[:2]
    if image_position is None:
        margin = 50
        origin_u = margin
        origin_v = h - margin
        image_position = (origin_u, origin_v)
    origin_2d = image_position

    # 定义 Base 坐标系下的轴点
    axis_points_base = np.array([
        [0, 0, 0],
        [axis_length, 0, 0],  # X
        [0, axis_length, 0],  # Y
        [0, 0, axis_length],  # Z
    ])

    # 变换到相机坐标系：R @ point（无平移）
    axis_points_cam = (R_cam_base @ axis_points_base.T).T

    # 使用固定深度 fake_z=1.0 进行比例投影（仅用于方向显示）
    def project_point(cam_pt, K, origin_2d):
        x, y, z = cam_pt
        fake_z = 1.0
        u = int(origin_2d[0] + (K[0, 0] * x / fake_z))
        v = int(origin_2d[1] - (K[1, 1] * y / fake_z))  # 注意：图像y向下，所以减
        return (u, v)

    colors = {'X': (255, 0, 0), 'Y': (0, 255, 0), 'Z': (0, 0, 255)}
    endpoints = []
    for i in range(1, 4):
        pt_cam = axis_points_cam[i]
        try:
            pt_2d = project_point(pt_cam, K, origin_2d)
            endpoints.append(pt_2d)
        except:
            endpoints.append(None)

    # 绘制箭头
    for (axis_name, color), end_2d in zip(colors.items(), endpoints):
        if end_2d is None:
            continue
        cv2.arrowedLine(image, origin_2d, end_2d, color, thickness=2, tipLength=0.2)

        # 轴标签 (X/Y/Z)
        offset = (5, -5)
        cv2.putText(image, axis_name,
                    (end_2d[0] + offset[0], end_2d[1] + offset[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1,
                    lineType=cv2.LINE_AA)

    # 绘制“Base”文字标签
    label_pos = (origin_2d[0] + 10, origin_2d[1] + 25)
    cv2.putText(image, "Base",
                label_pos,
                cv2.FONT_HERSHEY_SIMPLEX, label_scale, (0, 0, 0), 2,
                lineType=cv2.LINE_AA)  # 黑底轮廓
    cv2.putText(image, "Base",
                label_pos,
                cv2.FONT_HERSHEY_SIMPLEX, label_scale, (255, 255, 255), 1,
                lineType=cv2.LINE_AA)  # 白字

    return image


def draw_axis(camera, ur5, frame_rgb):
    """
    主函数：绘制 End-effector 坐标系（真实投影）和 Base 坐标系（固定左下角）
    """
    K = camera.get_intrinsic_matrix()
    T_base_to_camera = camera.get_extrinsic_matrix()
    T_camera_to_base = np.linalg.inv(T_base_to_camera)

    frame_draw = frame_rgb.copy()

    # --- 获取当前末端位姿（TCP 相对于 Base）---
    tcp_pose = ur5.get_tcp()  # 假设返回 [x, y, z, rx, ry, rz]
    x, y, z, rx, ry, rz = tcp_pose
    R_tcp, _ = cv2.Rodrigues(np.array([rx, ry, rz]))
    t_tcp = np.array([[x, y, z]]).T

    # 构造齐次变换矩阵：Base → TCP
    T_base_to_tcp = np.eye(4)
    T_base_to_tcp[:3, :3] = R_tcp
    T_base_to_tcp[:3, 3] = t_tcp.flatten()

    # --- 计算在相机坐标系下的位姿 ---

    # 1. End-effector (TCP) 在相机坐标系中
    T_camera_to_tcp = T_camera_to_base @ T_base_to_tcp
    R_cam_tcp = T_camera_to_tcp[:3, :3]
    t_cam_tcp = T_camera_to_tcp[:3, 3:4]

    # 2. Base 坐标系的旋转（用于固定绘制）
    R_cam_base = T_camera_to_base[:3, :3]

    # --- 绘制两个坐标系 ---

    # 绘制 End-effector：真实投影
    frame_draw = draw_axis_on_image(
        frame_draw, K, R_cam_tcp, t_cam_tcp,
        axis_length=0.07, min_z=0.1, label="End-effector"
    )

    # 绘制 Base：固定在左下角
    frame_draw = draw_fixed_base_axis(
        frame_draw, K, R_cam_base,
        image_position=None,        # 自动设为左下角
        axis_length=0.05,           # 更短
        label_scale=0.4             # 小字体
    )

    return frame_draw