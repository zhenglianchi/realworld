import numpy as np
from UR_Base import UR_BASE
from LMP import rotate_pose_local_axis
import time

# 初始化机械臂和相机
ur5 = UR_BASE("192.168.111.10")

print(ur5.get_tcp())
#new_rotation = rotate_pose_local_axis(rotation, axis='x', angle_deg=np.pi/2)
'''print(rotation)
print(new_rotation)
new_pose = np.concatenate((translation, new_rotation))

ur5.moveL(new_pose)'''
