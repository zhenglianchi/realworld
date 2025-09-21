import cv2
import numpy as np

def draw_axis_on_image(image, K, R_cam_tcp, t_cam_tcp, axis_length=0.05, label="End-effector", label_scale=0.4):
    """
    在图像上绘制 End-effector 坐标系 (投影到相机图像)
    :param image: 输入图像
    :param K: 相机内参矩阵 (3x3)
    :param R_cam_tcp: TCP 在相机坐标系下的旋转矩阵 (3x3)
    :param t_cam_tcp: TCP 在相机坐标系下的平移向量 (3,1)
    :param axis_length: 坐标轴长度
    :param label: 标签（默认 End-effector）
    :param label_scale: 字体缩放
    """
    # 定义 TCP 坐标系的原点和三个轴端点（局部坐标）
    axis_points = np.array([
        [0, 0, 0],
        [axis_length, 0, 0],
        [0, axis_length, 0],
        [0, 0, axis_length]
    ])  # (4,3)

    # 转换到相机坐标系
    axis_points_cam = (R_cam_tcp @ axis_points.T).T + t_cam_tcp.reshape(1, 3)

    # 投影到图像平面
    points_2d = []
    for pt in axis_points_cam:
        X, Y, Z = pt
        if Z <= 0:   # 在相机后方的点不投影
            points_2d.append(None)
            continue
        u = int((K[0, 0] * X / Z) + K[0, 2])
        v = int((K[1, 1] * Y / Z) + K[1, 2])
        points_2d.append((u, v))

    # 检查原点是否可见
    if points_2d[0] is None:
        return image

    origin_2d = points_2d[0]
    colors = {'X': (255, 0, 0), 'Y': (0, 255, 0), 'Z': (0, 0, 255)}

    # 绘制三根轴
    for axis_name, end_pt in zip(colors.keys(), points_2d[1:]):
        if end_pt is None:
            continue
        cv2.arrowedLine(image, origin_2d, end_pt, colors[axis_name], 1, tipLength=0.1)
        cv2.putText(image, axis_name, (end_pt[0]+5, end_pt[1]-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,255,255), 1, lineType=cv2.LINE_AA)

    # 绘制标签 (End-effector)
    cv2.putText(image, label, (origin_2d[0]+10, origin_2d[1]+25),
                cv2.FONT_HERSHEY_SIMPLEX, label_scale, (0,0,0), 2, lineType=cv2.LINE_AA)
    cv2.putText(image, label, (origin_2d[0]+10, origin_2d[1]+25),
                cv2.FONT_HERSHEY_SIMPLEX, label_scale, (255,255,255), 1, lineType=cv2.LINE_AA)

    return image




def draw_fixed_base_axis(image, K, R_cam_base, axis_length=0.05, label_scale=0.4):
    """
    在图像左下角区域使用真实透视投影绘制 Base 坐标系
    不再固定2D长度，而是将Base坐标系置于相机前方某3D位置后真实投影
    """
    h, w = image.shape[:2]

    # 在相机坐标系中为 Base 坐标系设置一个示意原点（调整此值可控制投影位置）
    t_cam_base_origin = np.array([-0.3, 0.3, 1.0])

    # 构造原点 + 三轴端点（在Base局部坐标系中）
    axis_points_local = np.array([
        [0, 0, 0],                  # Origin
        [axis_length, 0, 0],        # X axis
        [0, axis_length, 0],        # Y axis
        [0, 0, axis_length]         # Z axis
    ])

    # 转换到相机坐标系：先旋转，再平移
    axis_points_cam = (R_cam_base @ axis_points_local.T).T + t_cam_base_origin

    # 投影到图像平面
    points_2d = []
    for pt in axis_points_cam:
        X, Y, Z = pt
        if Z <= 1e-5:  # 避免无效投影
            points_2d.append(None)
            continue
        u = int(K[0, 0] * X / Z + K[0, 2])
        v = int(K[1, 1] * Y / Z + K[1, 2])
        # 可选：边界检查（防止绘制到图像外）
        if not (0 <= u < w and 0 <= v < h):
            points_2d.append(None)
            continue
        points_2d.append((u, v))

    origin_2d = points_2d[0]
    if origin_2d is None:
        return image

    colors = {'X': (255, 0, 0), 'Y': (0, 255, 0)}

    # 绘制三根轴（保持你原来的风格：细线、小箭头）
    for axis_name, end_pt in zip(colors.keys(), points_2d[1:]):
        if end_pt is None:
            continue
        cv2.arrowedLine(image, origin_2d, end_pt, colors[axis_name], 1, tipLength=0.1)
        cv2.putText(image, axis_name, (end_pt[0] + 5, end_pt[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, lineType=cv2.LINE_AA)

    # 绘制 "Base" 标签（保持你原来的双描边风格）
    cv2.putText(image, "Base", (origin_2d[0] + 10, origin_2d[1] + 25),
                cv2.FONT_HERSHEY_SIMPLEX, label_scale, (0, 0, 0), 2, lineType=cv2.LINE_AA)
    cv2.putText(image, "Base", (origin_2d[0] + 10, origin_2d[1] + 25),
                cv2.FONT_HERSHEY_SIMPLEX, label_scale, (255, 255, 255), 1, lineType=cv2.LINE_AA)

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
        axis_length=0.1, label="gripper",label_scale=0.3
    )

    '''# 绘制 Base：固定在左下角
    frame_draw = draw_fixed_base_axis(
        frame_draw, K, R_cam_base,
        axis_length=0.07,           # 更短
        label_scale=0.5             # 小字体
    )'''

    return frame_draw