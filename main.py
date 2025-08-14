from arguments import get_config
from interfaces import setup_LMP
from UR_Base import UR_BASE
from Threads import Update_State_Thread, Execute_Thread
from StateManager import StateManager
import threading
import time
import os
import time
import queue
import shutil 
import numpy as np
from Costmapvisualizer3D import VoxelSceneVisualizer


#load config
config_path = "configs/vlm_rlbench_config.yaml"
config = get_config(config_path=config_path)

state_manager = StateManager(config.json_path, poll_interval=0.02, log_interval=5)
state_manager.start_monitor()  # 启动自动监听

voxel_visualizer = VoxelSceneVisualizer(
    save_dir = config['visualizer'].save_dir
)

init_pose = np.array([
    -0.21573,
    -0.05917,
    0.60087,
    2.633,
    -1.579,
    -0.131
    ])
ur5 = UR_BASE("192.168.111.10",fisrt_tcp=init_pose)
time.sleep(6)

voxposer_ui, lmp_env = setup_LMP(config,ur5)

instruction = "Grasp the tape and place it on the mouse."

grasp_object = queue.Queue()
grasp_event = threading.Event()
init_grasp_finished = threading.Event()
finished_event = threading.Event()

def update_state(instruction,finished_event,grasp_event,grasp_object,state_manager,init_grasp_finished):
    lmp_env.update_mask_entities(instruction,finished_event,grasp_event,grasp_object,state_manager,init_grasp_finished)
    shutil.rmtree("tmp/images")
    shutil.rmtree("tmp/masks")

def run_voxposer_ui(instruction,lmp_env,finished_event,grasp_event,grasp_object,state_manager,voxel_visualizer,init_grasp_finished):
    voxposer_ui(instruction,lmp_env,grasp_event,grasp_object,state_manager,voxel_visualizer,init_grasp_finished)
    finished_event.set()


thread1 = Update_State_Thread(target=update_state, args=(instruction,finished_event,grasp_event,grasp_object,state_manager,init_grasp_finished))
thread2 = Execute_Thread(target=run_voxposer_ui, args=(instruction,lmp_env,finished_event,grasp_event,grasp_object,state_manager,voxel_visualizer,init_grasp_finished))

thread1.start()
thread2.start()
thread2.join()
thread1.join()
state_manager.stop_monitor()
finished_event.clear()
grasp_event.clear()


