from arguments import get_config
from interfaces import setup_LMP
from UR_Base import UR_BASE
from Threads import Update_State_Thread, Execute_Thread
from StateManager import StateManager
import threading
import time
import numpy as np
from Costmapvisualizer3D import VoxelSceneVisualizer
import queue

#load config
config_path = "configs/vlm_rlbench_config.yaml"
config = get_config(config_path=config_path)

state_manager = StateManager(config.json_path, poll_interval=0.02, log_interval=5)
state_manager.start_monitor()  # 启动自动监听

voxel_visualizer = VoxelSceneVisualizer(
    save_dir = config['visualizer'].save_dir
)

init_pose = np.array([   0.135,    -0.2622   ,   0.500,      3.02 ,   -0.12 ,   0.02])

ur5 = UR_BASE("192.168.111.10",fisrt_tcp=init_pose)
time.sleep(5)

voxposer_ui, lmp_env = setup_LMP(config,ur5)

instruction = "Open the drawer."

finished_event = threading.Event()
condition = threading.Condition()
image_share = queue.Queue()

def update_state(instruction,finished_event,state_manager,condition,image_share):
    lmp_env.update_mask_entities(instruction,finished_event,state_manager,condition,image_share)

def run_voxposer_ui(instruction,lmp_env,finished_event,state_manager,voxel_visualizer,image_share):
    voxposer_ui(instruction,lmp_env,state_manager,voxel_visualizer,image_share)
    finished_event.set()


thread1 = Update_State_Thread(target=update_state, args=(instruction,finished_event,state_manager,condition,image_share))
thread2 = Execute_Thread(target=run_voxposer_ui, args=(instruction,lmp_env,finished_event,state_manager,voxel_visualizer,image_share))

thread1.start()
with condition:
    condition.wait()
thread2.start()
thread2.join()
thread1.join()
state_manager.stop_monitor()
finished_event.clear()


