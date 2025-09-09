from arguments import get_config
from interfaces import setup_LMP
from UR_Base import UR_BASE
from Threads import Update_State_Thread, Execute_Thread
from StateManager import StateManager
import threading
import time
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

init_pose = np.array([   -0.29255,    -0.16637   ,   0.35149,      0.9136 ,   -3.00 ,   0.0874])

ur5 = UR_BASE("192.168.111.10",fisrt_tcp=init_pose)
time.sleep(5)

voxposer_ui, lmp_env = setup_LMP(config,ur5)

instruction = "Flower arranging task. Please insert the flower into the vase."

finished_event = threading.Event()
condition = threading.Condition()

def update_state(instruction,finished_event,state_manager,condition):
    lmp_env.update_mask_entities(instruction,finished_event,state_manager,condition)

def run_voxposer_ui(instruction,lmp_env,finished_event,state_manager,voxel_visualizer,):
    voxposer_ui(instruction,lmp_env,state_manager,voxel_visualizer,)
    finished_event.set()


thread1 = Update_State_Thread(target=update_state, args=(instruction,finished_event,state_manager,condition))
thread2 = Execute_Thread(target=run_voxposer_ui, args=(instruction,lmp_env,finished_event,state_manager,voxel_visualizer,))

thread1.start()
with condition:
    condition.wait()
thread2.start()
thread2.join()
thread1.join()
state_manager.stop_monitor()
finished_event.clear()


