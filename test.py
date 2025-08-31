import numpy as np
from UR_Base import UR_BASE
from LMP import rotate_pose_local_axis
import time

# 初始化机械臂和相机
'''init_pose = np.array([
    -0.425, -0.227, 0.432,
    1.494, -0.456, -0.683
])'''
ur5 = UR_BASE("192.168.111.10")
time.sleep(3)

translation = ur5.get_tcp()[:3]
rotation = ur5.get_tcp()[3:]

new_rotation = rotate_pose_local_axis(rotation, axis='z', angle_deg=-np.pi/2)
print(rotation)
print(new_rotation)
new_pose = np.concatenate((translation, new_rotation))

ur5.moveL(new_pose)
