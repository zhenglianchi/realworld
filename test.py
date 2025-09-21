'''from UR_Base import UR_BASE
import numpy as np
from LMP import rotate_pose_local_axis
import time

init_pose = np.array([   0.135,    -0.2622   ,   0.3611,      3.02 ,   -0.12 ,   0.02])

ur5 = UR_BASE("192.168.111.10",fisrt_tcp=init_pose)
time.sleep(5)
exit()
current_pose = ur5.get_tcp()
rot = current_pose[3:]
rot_new = rotate_pose_local_axis(rot, axis='y', angle_deg=-90, degrees=True)
rot_new = rotate_pose_local_axis(rot_new, axis='z', angle_deg=-90, degrees=True).tolist()
new_pos = np.array(current_pose[:3].tolist() + rot_new)
ur5.servoL(new_pos,time=10)

'''

tokens = "A1:\n(A) 15\nexpalnnation: rotate around z axis -15 degrees\n(B) keep the same\n"

answer = tokens.split("\n")[1].strip()
print(int(answer[3:]))