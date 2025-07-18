from arguments import get_config
from interfaces import setup_LMP
from UR_Base import UR_BASE
from Threads import Update_State_Thread, Execute_Thread
import threading
import time
import os
import time
import queue
import shutil 
import numpy as np


#load config
config_path = "configs/vlm_rlbench_config.yaml"
config = get_config(config_path=config_path)

init_pose = np.array([
    -0.31562,
    -0.17153,
    0.57808,
    -1.0583,
    -2.9459,
    -0.03206
    ])
ur5 = UR_BASE("192.168.111.10",fisrt_tcp=init_pose)

voxposer_ui, lmp_env = setup_LMP(config,ur5)

#instruction = input("请输入指令")
instruction = "only grasp the mouse"

# 创建锁
file_lock = threading.Lock()
grasp_object = queue.Queue()
grasp_event = threading.Event()
finished_event = threading.Event()

def update_state(instruction,file_lock,finished_event,grasp_event,grasp_object):
    lmp_env.update_mask_entities(instruction,file_lock,finished_event,grasp_event,grasp_object)
    shutil.rmtree("tmp/images")
    shutil.rmtree("tmp/masks")
    os.remove(config["json_path"])

def run_voxposer_ui(instruction,file_lock,lmp_env,finished_event,grasp_event,grasp_object):
    voxposer_ui(instruction,file_lock,lmp_env,grasp_event,grasp_object)
    finished_event.set()


thread1 = Update_State_Thread(target=update_state, args=(instruction,file_lock,finished_event,grasp_event,grasp_object))
thread2 = Execute_Thread(target=run_voxposer_ui, args=(instruction,file_lock,lmp_env,finished_event,grasp_event,grasp_object))

thread1.start()
while not os.path.exists(config["json_path"]):
    time.sleep(1)

thread2.start()
thread2.join()
thread1.join()
finished_event.clear()
grasp_event.clear()

