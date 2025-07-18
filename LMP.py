from openai import OpenAI
from utils import load_prompt,normalize_vector,bcolors,get_clock_time
from VLM_demo import encode_image,read_state
import json
import os
import numpy as np
from transforms3d.euler import euler2quat,quat2euler
from transforms3d.quaternions import qinverse,qmult
from scipy.spatial.transform import Rotation as R
import queue
from scipy.ndimage import distance_transform_edt
from Threads import Low_Execute_Thread,traj_Thread,map_Thread
import threading
import time

# creating some aliases for end effector and table in case LLMs refer to them differently (but rarely this happens)
EE_ALIAS = ['ee', 'endeffector', 'end_effector', 'end effector', 'gripper', 'hand']
TABLE_ALIAS = ['table', 'desk', 'workstation', 'work_station', 'work station', 'workspace', 'work_space', 'work space']


class LMP:
    """Language Model Program (LMP), adopted from Code as Policies."""
    def __init__(self, name, cfg, debug=False, env='rlbench'):
        self._name = name
        self._cfg = cfg
        self._debug = debug
        self._planner_prompt = load_prompt(f"{env}/{self._cfg['planner_prompt_fname']}.txt")
        self._action_state_prompt = load_prompt(f"{env}/{self._cfg['vision_prompt_fname']}.txt")

        self._stop_tokens = [self._cfg['stop']]
        self._context = None
        self.mask_path = "./tmp/masks/"
        self.image_path = "./tmp/images/"
        self.state_json_path = "./tmp/state.json"
        #set your api_key Qwen
        self.api_key= "sk-2b726a0c6b6a4554b7834df6bac0b803"
        self.base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"

        self.shared_queue = queue.Queue()
        self.quat_queue = queue.Queue()

        self.movable_var = None
        self.affordable_map = None
        self.avoidance_map = None
        self.rotation_map = None
        self.velocity_map = None
        self.gripper_map = None

        self.move = None

    def get_last_filename(self,folder):
        while True:
            filenames = os.listdir(folder)
            if len(filenames) != 0:
                filename = filenames[-1]
                return f"{folder}{filename}"
            else:
                time.sleep(1)


    def generate_planning(self, query):
        user_query = f'{self._cfg["query_prefix"]}{query}{self._cfg["query_suffix"]}'

        planner_prompt = self._planner_prompt

        if self._context :
            user_query = f"# Objects : {self._context}\n" + user_query

        #print(user_query)

        client = OpenAI(api_key=self.api_key,base_url=self.base_url)
        
        filepath = self.get_last_filename(self.mask_path)
        base64_image = encode_image(filepath)

        completion = client.chat.completions.create(
            model=self._cfg['vision_model'],
            messages=[{"role": "user","content": [
                {"type": "text","text": f"This is a robotic arm operation scene image.\n{planner_prompt}\nThe above are some examples of planning, please give the corresponding planning according to the image I gave you next:\n{user_query}. The output format likely is\n" + "planner : ['', '', '', '']\nOther than that, don't give me any superfluous information and hints.The objects in the generated plan should match the names in the given image"},
                {"type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}, 
                }
                ]}],
        )

        planner = completion.choices[0].message.content

        planning = json.loads(planner.split(":")[-1].strip())

        return planning
    
    def get_state(self, state_json_path, lock):
      state = read_state(state_json_path, lock)
      return state

    def _vlmapi_call(self,image_path, query, planner ,action, objects):
        client = OpenAI(api_key=self.api_key,base_url=self.base_url)

        base64_image = encode_image(image_path)

        prompt = self._action_state_prompt

        completion = client.chat.completions.create(
            model=self._cfg['vision_model'],  
            messages=[{"role": "user","content": [
                    {"type": "text","text": f"This is a robotic arm operation scene." + f"The format of output should be like {prompt}.\n Objects : {objects}\nMoves : [grasp,move],\nQuery : {query}\nPlanner : {planner}\nAction : {action}\nPlease just give me the corresponding json, no explanation and no text required"},
                    {"type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}, 
                    }
                    ]}]
            )

        resstr = completion.choices[0].message.content.replace("```","").replace("json","")

        state = json.loads(resstr)

        return state
    
    def get_next_valid_waypoint(self, curr_xyz):
        queue_list = list(self.shared_queue.queue)
        min_dist = float('inf')
        closest_idx = -1
        for idx, wp in enumerate(queue_list):
            dist = np.linalg.norm(curr_xyz - wp[0])
            if dist < min_dist:
                min_dist = dist
                closest_idx = idx
        
        for i in range(closest_idx+1):
            self.shared_queue.get()

    def get_map(self, map_lock):
        with map_lock:
            return self.movable_var, self.affordable_map, self.avoidance_map, self.rotation_map, self.velocity_map, self.gripper_map


    def __get__affordable_map(self,action_state,lmp_env,object_state,grasp_event,grasp_object):
        affordable_map = None
        affordable = action_state["affordable"]
        affordable_set = affordable["set"]
        if affordable_set != "default" :
            move_mode = affordable["move"]
            affordable_map = lmp_env._get_default_voxel_map('target')()
            affordable_var = affordable["object"]
            object = object_state[affordable_var]["obs"]
            if move_mode == "move":
                center_x, center_y, center_z = eval(affordable["center_x, center_y, center_z"])
                (min_x, min_y, min_z), (max_x, max_y, max_z) = eval(affordable["(min_x, min_y, min_z), (max_x, max_y, max_z)"])
            if move_mode == "grasp":
                translation = eval(affordable["translation"])
            x = eval(affordable["x"])
            y = eval(affordable["y"])
            z = eval(affordable["z"])
            target_affordance = affordable["target_affordance"]
            x,y,z = lmp_env._world_to_voxel(np.array([x,y,z]))
            affordable_map[x,y,z] = target_affordance
        return affordable_map
    
    def __get__avoidance_map(self,action_state,lmp_env,object_state):
        avoidance_map = lmp_env._get_default_voxel_map('obstacle')()
        avoidance = action_state["avoid"]
        avoidance_set = avoidance["set"]
        if avoidance_set != "default" :
            avoidance_var = action_state["avoid"]["object"]
            if avoidance_var not in object_state.keys():
                print(f"Object {avoidance_var} not found in scene in this step.")
                pass
            object = object_state[avoidance_var]["obs"]
            center_x, center_y, center_z = eval(avoidance["center_x, center_y, center_z"])
            (min_x, min_y, min_z), (max_x, max_y, max_z) = eval(avoidance["(min_x, min_y, min_z), (max_x, max_y, max_z)"])
            x = eval(avoidance["x"])
            y = eval(avoidance["y"])
            z = eval(avoidance["z"])
            radius_cm = avoidance["radius_cm"]
            value = avoidance["value"]
            avoidance_map = set_voxel_by_radius(avoidance_map, [x,y,z], radius_cm, value)
        return avoidance_map
    
    def __get__gripper_map(self,action_state,lmp_env,object_state):
        gripper_map = lmp_env._get_default_voxel_map('gripper')()
        gripper = action_state["gripper"]
        gripper_set = gripper["set"]
        if gripper_set != "default" :
            if "object" not in action_state["gripper"].keys():
                gripper_map[:, :, :] = 1
                return gripper_map
            gripper_var = action_state["gripper"]["object"]
            object = object_state[gripper_var]["obs"]
            center_x, center_y, center_z = eval(gripper["center_x, center_y, center_z"])
            (min_x, min_y, min_z), (max_x, max_y, max_z) = eval(gripper["(min_x, min_y, min_z), (max_x, max_y, max_z)"])
            x = eval(gripper["x"])
            y = eval(gripper["y"])
            z = eval(gripper["z"])
            radius_cm = gripper["radius_cm"]
            value = gripper["value"]
            gripper_map = set_voxel_by_radius(gripper_map, [x,y,z], radius_cm, value)
        return gripper_map
    
    
    def __get__rotation_map(self,action_state,lmp_env,object_state):
        rotation_map = lmp_env._get_default_voxel_map('rotation')()

        return rotation_map
    
    def __get__velocity_map(self,action_state,lmp_env,object_state):
        velocity_map = lmp_env._get_default_voxel_map('velocity')()
        velocity = action_state["velocity"]
        velocity_set = velocity["set"]
        if velocity_set != "default" :
            target_velocity = velocity["target_velocity"]
            velocity_map[:] = target_velocity
        return velocity_map

    def init_map(self, action_state):
        affordable = action_state["affordable"]
        affordable_set = affordable["set"]
        if affordable_set != "default" :
            self.move = True
        else:
            self.move = False




    def __thread_update_map(self, lmp_env, action_state, file_lock, update_stop_event, exec_stop_event, map_lock, grasp_event, grasp_object):
        global _map_size, _resolution
        _map_size = lmp_env._map_size
        _resolution = lmp_env._resolution
        while not update_stop_event.is_set():
            start_time = time.time()

            object_state = self.get_state(self.state_json_path,file_lock)

            affordable_map = self.__get__affordable_map(action_state,lmp_env,object_state,grasp_event,grasp_object)
            rotation_map = self.__get__rotation_map(action_state,lmp_env,object_state)
            velocity_map = self.__get__velocity_map(action_state,lmp_env,object_state)
            gripper_map = self.__get__gripper_map(action_state,lmp_env,object_state)
            avoidance_map = self.__get__avoidance_map(action_state,lmp_env,object_state)

            movable = action_state["movable"]
            movable_var = object_state[movable]["obs"]

            with map_lock:
                self.movable_var = movable_var
                self.affordable_map = affordable_map
                self.avoidance_map = avoidance_map
                self.rotation_map = rotation_map
                self.velocity_map = velocity_map
                self.gripper_map = gripper_map

            end_time = time.time()
            print(f"{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] updated map in {end_time - start_time:.3f}s{bcolors.ENDC}")


            
    def __thread_update_traj(self, lmp_env, action_state, file_lock, update_stop_event, exec_stop_event, map_lock):
        while not update_stop_event.is_set():
            start_time = time.time()
            movable_var, affordance_map, avoidance_map, rotation_map, velocity_map, gripper_map = self.get_map(map_lock)
            if self.move:
                # Preprocess avoidance map
                _avoidance_map = lmp_env._preprocess_avoidance_map(avoidance_map, affordance_map, movable_var)

                start_pos = lmp_env.get_ee_pos().copy()  # 直接获取实时位置
                
                # Optimize path and log
                path_voxel, planner_info = lmp_env._planner.optimize(start_pos, affordance_map, _avoidance_map)
                assert len(path_voxel) > 0, 'path_voxel is empty'

                print(path_voxel)
                
                trajectory = []
                # Convert voxel path to world trajectory, and include rotation, velocity, and gripper information
                for i in range(len(path_voxel)):
                    voxel_xyz = path_voxel[i]
                    world_xyz = lmp_env._voxel_to_world(voxel_xyz)
                    voxel_xyz = np.round(voxel_xyz).astype(int)

                    rotation = lmp_env.ur5.get_tcp()[3:]
                    rotation_map[voxel_xyz[0], voxel_xyz[1], voxel_xyz[2]] = rotation
                    
                    velocity = velocity_map[voxel_xyz[0], voxel_xyz[1], voxel_xyz[2]]
                    gripper = gripper_map[voxel_xyz[0], voxel_xyz[1], voxel_xyz[2]]
                    
                    if (i == len(path_voxel) - 1) and not (np.all(gripper_map == 1) or np.all(gripper_map == 0)):
                        less_common_value = 1 if np.sum(gripper_map == 1) < np.sum(gripper_map == 0) else 0
                        less_common_indices = np.where(gripper_map == less_common_value)
                        less_common_indices = np.array(less_common_indices).T
                        closest_distance = np.min(np.linalg.norm(less_common_indices - voxel_xyz[None, :], axis=0))
                        if closest_distance <= 3:
                            gripper = less_common_value
                    
                    trajectory.append((world_xyz, rotation, velocity, gripper))

                # Clear old queue and insert new trajectory
                while not self.shared_queue.empty():
                    try:
                        self.shared_queue.get_nowait()
                    except Exception:
                        break
                for wp in trajectory:
                    self.shared_queue.put(wp)

                end_time = time.time()
                print(f"{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] updated trajectory in {end_time - start_time:.3f}s{bcolors.ENDC}")
            else:
                print("Gripper manipulation, no need to update traj")
                break


    def __thread_execute_traj(self, lmp_env, action_state, file_lock, update_stop_event, exec_stop_event, map_lock):
        if self.move:
            i = 0
            while not exec_stop_event.is_set():
                # 这里后期可以优化
                movable_var, affordable_map, avoidance_map, rotation_map, velocity_map, gripper_map = self.get_map(map_lock)
                if self.shared_queue.empty():
                    time.sleep(0.1)
                    continue
                queue_list = list(self.shared_queue.queue)

                curr_xyz = movable_var['_position_world']
                self.get_next_valid_waypoint(curr_xyz)

                waypoint = self.shared_queue.get()

                # check if the movement is finished
                if np.linalg.norm(movable_var['_position_world'] - queue_list[-1][0]) <= 0.02:
                    print(f"{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] reached last waypoint; curr_xyz={movable_var['_position_world']}, target={queue_list[-1][0]} (distance: {np.linalg.norm(movable_var['_position_world'] - queue_list[-1][0]):.3f})){bcolors.ENDC}")
                    exec_stop_event.set()
                    update_stop_event.set()
                    '''
                    令末端到达最后一个点
                    这里可以考虑用强化学习来训练一个模型去预测抓取适配器的位置
                    '''
                    '''
                    ee_pos_world = queue_list[-1][0]
                    ee_rot_world = queue_list[-1][1]
                    ee_pose_world = np.concatenate([ee_pos_world, ee_rot_world])
                    ee_speed = queue_list[-1][2]
                    gripper_state = queue_list[-1][3]
                    lmp_env.ur5.apply_action(np.concatenate([ee_pose_world, [gripper_state]]))
                    '''
                    break

                # execute waypoint
                controller_info = lmp_env.ur5.execute(movable_var, waypoint)

                dist2target = np.linalg.norm(movable_var['_position_world'] - queue_list[-1][0])
                print(f'{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] completed waypoint {i+1} (wp: {waypoint[0].round(3)}, actual: {movable_var["_position_world"].round(3)}, target: {queue_list[-1][0].round(3)}, start: {queue_list[0][0].round(3)}, dist2target: {dist2target.round(3)}){bcolors.ENDC}')
                
                i += 1
            print(f'{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] finished executing path via controller{bcolors.ENDC}')


    def __call__(self, query, file_lock, lmp_env, grasp_event, grasp_object):
        planning = self.generate_planning(query)
        print(planning)
        planning_ = planning.copy()
        update_stop_event = threading.Event()
        exec_stop_event = threading.Event()
        map_lock = threading.Lock()
        while len(planning) >= 0:
            action = planning.pop(0)
            action_state = None
            filenames = os.listdir("cache")
            for filename in filenames:
                action_temp = json.load(open(os.path.join("cache", filename), "r"))
                if action_temp["Action"] == action:
                    print(f"using cache {filename}")
                    action_state = action_temp
                    break
            
            # 如果没有缓存，则调用API获取动作状态
            if action_state is None:
                filepath = self.get_last_filename(self.mask_path)
                action_state  = self._vlmapi_call(filepath, query=query, planner=planning_, action=action, objects=self._context)
                current_action = action_state["Action"]
                with open(f"./cache/{current_action}.json", 'w', encoding='utf-8') as json_file:
                    json.dump(action_state, json_file)

            print(action_state)

            affordable_var = action_state["affordable"]
            move = affordable_var["move"]
            if move == "grasp":
                grasp_event.set()
                object_name = affordable_var["object"]
                print("抓取物体",object_name)
                grasp_object.put(object_name)

            self.init_map(action_state)

            time.sleep(5)

            # 启动更新路径的线程
            map_thread = map_Thread(target=self.__thread_update_map, args=(lmp_env, action_state, file_lock, update_stop_event,exec_stop_event,map_lock, grasp_event, grasp_object,))
            map_thread.daemon = True  # 设置为守护线程，随主线程退出
            map_thread.start()

            time.sleep(5)

            # 启动更新路径的线程
            traj_thread = traj_Thread(target=self.__thread_update_traj, args=(lmp_env, action_state, file_lock, update_stop_event,exec_stop_event,map_lock, ))
            traj_thread.daemon = True  # 设置为守护线程，随主线程退出
            traj_thread.start()

            time.sleep(5)

            # 启动执行路径的线程
            execute_thread = Low_Execute_Thread(target=self.__thread_execute_traj, args=(lmp_env, action_state, file_lock, update_stop_event,exec_stop_event,map_lock, ))
            execute_thread.daemon = True  # 设置为守护线程，随主线程退出
            execute_thread.start()

            execute_thread.join()
            traj_thread.join()
            map_thread.join()

            update_stop_event.clear()
            exec_stop_event.clear()
            grasp_event.clear()
            grasp_object.get_nowait()

            # Clear old queue and insert new trajectory
            while not self.shared_queue.empty():
                try:
                    self.shared_queue.get_nowait()
                except Exception:
                    break

            if len(planning) == 0:
                print(f"{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] finished all planning{bcolors.ENDC}")
                time.sleep(1)
                lmp_env.ur5.reset_to_default_pose()
                break



def merge_dicts(dicts):
    return {
        k : v 
        for d in dicts
        for k, v in d.items()
    }
    

def exec_safe(code_str, gvars=None, lvars=None):
    banned_phrases = ['import', '__']
    for phrase in banned_phrases:
        assert phrase not in code_str
  
    if gvars is None:
        gvars = {}
    if lvars is None:
        lvars = {}
    empty_fn = lambda *args, **kwargs: None
    custom_gvars = merge_dicts([
        gvars,
        {'exec': empty_fn, 'eval': empty_fn}
    ])
    try:
        exec(code_str, custom_gvars, lvars)
    except Exception as e:
        print(f'Error executing code:\n{code_str}')
        raise e


def cm2index(cm, direction):
    global _map_size, _resolution
    if isinstance(direction, str) and direction == 'x':
      x_resolution = _resolution[0] * 100  # resolution is in m, we need cm
      return int(cm / x_resolution)
    elif isinstance(direction, str) and direction == 'y':
      y_resolution = _resolution[1] * 100
      return int(cm / y_resolution)
    elif isinstance(direction, str) and direction == 'z':
      z_resolution = _resolution[2] * 100
      return int(cm / z_resolution)
    else:
      # calculate index along the direction
      assert isinstance(direction, np.ndarray) and direction.shape == (3,)
      direction = normalize_vector(direction)
      x_cm = cm * direction[0]
      y_cm = cm * direction[1]
      z_cm = cm * direction[2]
      x_index = cm2index(x_cm, 'x')
      y_index = cm2index(y_cm, 'y')
      z_index = cm2index(z_cm, 'z')
      return np.array([x_index, y_index, z_index])
  
def index2cm(index, direction=None):
    global _map_size, _resolution
    if direction is None:
      average_resolution = np.mean(_resolution)
      return index * average_resolution * 100  # resolution is in m, we need cm
    elif direction == 'x':
      x_resolution = _resolution[0] * 100
      return index * x_resolution
    elif direction == 'y':
      y_resolution = _resolution[1] * 100
      return index * y_resolution
    elif direction == 'z':
      z_resolution = _resolution[2] * 100
      return index * z_resolution
    else:
      raise NotImplementedError
    
def pointat2quat(vector):
    assert isinstance(vector, np.ndarray) and vector.shape == (3,), f'vector: {vector}'
    return pointat2quat(vector)    # append the last waypoint a few more times for the robot to stabilize

def vec2quat(vec):
    vec = vec / np.linalg.norm(vec)
    # 目标方向是z轴
    target = np.array([0, 0, 1])
    # 使用Rotation.from_rotvec来获取从z轴到v的旋转
    rotation = R.align_vectors([vec], [target])[0]
    # 获取四元数
    quat = rotation.as_quat()  # 返回四元数 [x, y, z, w] 格式
    return quat


def set_voxel_by_radius(voxel_map, voxel_xyz, radius_cm=0, value=1):
    """given a 3D np array, set the value of the voxel at voxel_xyz to value. If radius is specified, set the value of all voxels within the radius to value."""
    global _map_size, _resolution
    voxel_map[voxel_xyz[0], voxel_xyz[1], voxel_xyz[2]] = value
    if radius_cm > 0:
      radius_x = cm2index(radius_cm, 'x')
      radius_y = cm2index(radius_cm, 'y')
      radius_z = cm2index(radius_cm, 'z')
      # simplified version - use rectangle instead of circle (because it is faster)
      min_x = max(0, voxel_xyz[0] - radius_x)
      max_x = min(_map_size, voxel_xyz[0] + radius_x + 1)
      min_y = max(0, voxel_xyz[1] - radius_y)
      max_y = min(_map_size, voxel_xyz[1] + radius_y + 1)
      min_z = max(0, voxel_xyz[2] - radius_z)
      max_z = min(_map_size, voxel_xyz[2] + radius_z + 1)
      voxel_map[min_x:max_x, min_y:max_y, min_z:max_z] = value
    return voxel_map
