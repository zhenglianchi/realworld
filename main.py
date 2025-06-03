from arguments import get_config
from interfaces import setup_LMP
from UR_Base import UR_BASE
import threading
import time
import os
import time
import queue
import shutil 


#load config
config_path = "configs/vlm_rlbench_config.yaml"
config = get_config(config_path=config_path)

init_pose = [0,0,0,0,0,0]
ur5 = UR_BASE("192.168.111.10",fisrt_tcp=init_pose)

voxposer_ui, lmp_env = setup_LMP(config,ur5)

instruction = input("请输入指令")

# 创建锁
file_lock = threading.Lock()
q = queue.Queue()

def update_state(file_lock,q):
    lmp_env.update_mask_entities(file_lock,q)
    shutil.rmtree("tmp/images")
    shutil.rmtree("tmp/masks")
    os.remove(config["json_path"])

def run_voxposer_ui(instruction,file_lock,lmp_env,q):
    voxposer_ui(instruction,file_lock,lmp_env)
    q.put(0)


thread1 = threading.Thread(target=update_state, args=(file_lock,q,))
thread2 = threading.Thread(target=run_voxposer_ui, args=(instruction,file_lock,lmp_env,q,))

thread1.start()
while not os.path.exists(config["json_path"]):
    time.sleep(1)


#thread2.start()
#thread2.join()
thread1.join()



