from UR_Base import UR_BASE
import numpy as np

init_pose = np.array([
    -0.14746,
    -0.108,
    0.53696,
    0.512,
    3.046,
    -0.003
    ])

ur5 = UR_BASE("192.168.111.10",fisrt_tcp=init_pose)