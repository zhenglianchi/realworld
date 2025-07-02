import rtde_control
import rtde_receive
from scipy.spatial.transform import Rotation as R
import numpy as np

class UR_BASE(object):
    def __init__(self, HOST, fisrt_tcp=None):
        self.hostname = HOST
        self.rtde_c = rtde_control.RTDEControlInterface(HOST)
        self.rtde_r = rtde_receive.RTDEReceiveInterface(HOST)
        self.current_tcp = self.get_tcp()
        self.workspace_bounds_min = np.array([-0.27499999, -0.65500004,  0.75199986])
        self.workspace_bounds_max = np.array([0.77499999, 0.65500004, 1.75199986])
        if fisrt_tcp is not None:
            self.init_pose = fisrt_tcp
            self.moveL(fisrt_tcp)

    def reset_to_default_pose(self):
        self.moveL(self.init_pose)
        
    def set_digital_out(self, num, bool):
        self.rob.set_digital_out(num, bool)
        self.rtde_c.disconnect()
        self.rtde_c.reconnect()

    def move_joint_path(self, path):
        # speed = 0.9
        # acc = 0.5
        # blend = 0.02

        joints = [0, 0, 0, 0, 0, 0]
        for i in range(len(path)):
            for n in range(6):
                joints[n] = path[i][n] * np.pi / 180

            self.rtde_c.moveJ(joints, 2, 1.5, True)  # 代表函数不阻塞，可以进行改变，从而实现控制

    def stop_robot(self):
        self.rtde_c.servoStop()
        self.rtde_c.stopScript()

    def disconnect(self):
        self.rtde_c.servoStop()
        self.rtde_c.stopScript()
        self.rtde_c.disconnect()

    # 工具的笛卡尔坐标(x,y,z,rx,ry,rz)
    def get_tcp(self):
        return np.array(self.rtde_r.getActualTCPPose(),dtype=np.float32)

    def get_joint(self):
        return self.rtde_r.getActualQ()

    def moveL(self, pose, asy=True, speed=0.02, acc=0.2):
        pose = pose.tolist()
        self.rtde_c.moveL(pose, speed, acc, asy)

    def speedJ(self, joint_speed, acc=0.25, control_period=0.02):
        joint_speed = joint_speed.tolist()
        self.rtde_c.speedJ(joint_speed, acc, control_period)
        
    def stopSpeedJ(self):
        self.rtde_c.speedStop()

    def speedL(self, ee_speed, acc=0.25, control_period=0.02):
        self.rtde_c.speedL(ee_speed, acc, control_period)

    def servoL(self, pose, speed=0.01, acc=0.02):
        pose = pose.tolist()
        self.rtde_c.servoL(pose, speed, acc, 0.01, 0.05, 300)

    def execute(self,movable_var, waypoint):
        print(movable_var,waypoint)

    def ur_pose_to_matrix(self):
        """
        将 UR5 的 [x,y,z,Rx,Ry,Rz] 格式转换为 4x4 变换矩阵
        
        参数:
        - pose: list 或 numpy array，长度为6，顺序是 [x, y, z, Rx, Ry, Rz]
        
        返回:
        - T: 4x4 的变换矩阵，表示从工具坐标系到基座坐标系的变换
        """
        x, y, z, rx, ry, rz = self.get_tcp()
        
        # 平移部分
        translation = np.array([x, y, z])
        
        # 旋转部分：UR 中的 Rx,Ry,Rz 是 axis-angle 表示法（单位：弧度）
        rotation = R.from_rotvec([rx, ry, rz]).as_matrix()
        
        # 构建齐次变换矩阵
        T = np.eye(4)
        T[:3, :3] = rotation
        T[:3, 3] = translation
        
        return T