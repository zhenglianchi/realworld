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
        self.state_json_path = "./tmp/state_front.json"
        #set your api_key Qwen
        self.api_key= "sk-2b726a0c6b6a4554b7834df6bac0b803"
        self.base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"

        self.shared_queue = queue.Queue()
        self.quat_queue = queue.Queue()

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

        print(user_query)

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
    
    def get_state(self, state_json_path,lock):
      state = read_state(state_json_path,lock)
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


    def __get__affordable_map(self,action_state,lmp_env,object_state):
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
        rotation = action_state["rotation"]
        rotation_set = rotation["set"]
        if rotation_set != "default" and self.quat_queue.empty():
            rotation_var = action_state["rotation"]["object"]
            if rotation_var not in object_state.keys():
                print(f"Object {rotation_var} not found in scene in this step.")
                pass
            object = object_state[rotation_var]["obs"]
            target_rotation = eval(rotation["target_rotation"])
            current_rotation = lmp_env._env.get_ee_quat()
            quat_traj = lmp_env.interpolate_quaternions(current_rotation.tolist(), target_rotation.tolist(), 9)
            #rotation_map[:, :, :] = target_rotation
            '''for wp in quat_traj:
                self.quat_queue.put(wp)'''

        return rotation_map
    
    def __get__velocity_map(self,action_state,lmp_env,object_state):
        velocity_map = lmp_env._get_default_voxel_map('velocity')()
        velocity = action_state["velocity"]
        velocity_set = velocity["set"]
        if velocity_set != "default" :
            target_velocity = velocity["target_velocity"]
            velocity_map[:] = target_velocity
        return velocity_map


    def get_update_map(self, action_state, lock, lmp_env):
        global _map_size, _resolution
        _map_size = lmp_env._map_size
        _resolution = lmp_env._resolution

        object_state = self.get_state(self.state_json_path,lock)

        affordable_map = self.__get__affordable_map(action_state,lmp_env,object_state)
        rotation_map = self.__get__rotation_map(action_state,lmp_env,object_state)
        velocity_map = self.__get__velocity_map(action_state,lmp_env,object_state)
        gripper_map = self.__get__gripper_map(action_state,lmp_env,object_state)
        avoidance_map = self.__get__avoidance_map(action_state,lmp_env,object_state)

        movable = action_state["movable"]
        #movable_var = object_state[movable]["obs"]
        movable_var = lmp_env.get_ee_obs()["obs"]
        object_centric = (not movable_var['name'] in EE_ALIAS)

        return movable_var, affordable_map, avoidance_map, rotation_map, velocity_map, gripper_map, object_centric
            
    def __thread_update_traj(self, lmp_env, action_state, file_lock, update_stop_event, exec_stop_event):
        while not update_stop_event.is_set():
            start_time = time.time()
            movable_var, affordance_map, avoidance_map, rotation_map, velocity_map, gripper_map, object_centric = self.get_update_map(action_state, file_lock, lmp_env)
            if affordance_map is not None:
                # Preprocess avoidance map
                _avoidance_map = lmp_env._preprocess_avoidance_map(avoidance_map, affordance_map, movable_var)

                start_pos = lmp_env.get_ee_pos().copy()  # 直接获取实时位置
                
                # Optimize path and log
                path_voxel, planner_info = lmp_env._planner.optimize(start_pos, affordance_map, _avoidance_map,
                                                                    object_centric=object_centric)
                assert len(path_voxel) > 0, 'path_voxel is empty'
                
                trajectory = []
                # Convert voxel path to world trajectory, and include rotation, velocity, and gripper information
                for i in range(len(path_voxel)):
                    voxel_xyz = path_voxel[i]
                    world_xyz = lmp_env._voxel_to_world(voxel_xyz)
                    voxel_xyz = np.round(voxel_xyz).astype(int)
                    if not self.quat_queue.empty():
                        rotation = self.quat_queue.get().tolist()
                        rotation_map[voxel_xyz[0], voxel_xyz[1], voxel_xyz[2]] = rotation
                    else:
                        rotation = lmp_env._env.get_ee_quat()
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

                # For stability, repeat the last waypoint a few more times
                for _ in range(2):
                    trajectory.append(trajectory[-1])

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


    def __thread_execute_traj(self, lmp_env, action_state, file_lock, update_stop_event, exec_stop_event):
        movable_var, affordable_map, avoidance_map, rotation_map, velocity_map, gripper_map, object_centric = self.get_update_map(action_state, file_lock, lmp_env)
        if affordable_map is not None:
            i = 0
            while not exec_stop_event.is_set():
                movable_var, affordable_map, avoidance_map, rotation_map, velocity_map, gripper_map, object_centric = self.get_update_map(action_state, file_lock, lmp_env)
                if self.shared_queue.empty():
                    time.sleep(0.2)
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
                    break
                # skip waypoint if moving to this point is going in opposite direction of the final target point
                # (for example, if you have over-pushed an object, no need to move back)
                if i != 0 and i != len(queue_list) - 1:
                    movable2target = queue_list[-1][0] - movable_var['_position_world']
                    movable2waypoint = waypoint[0] - movable_var['_position_world']
                    if np.dot(movable2target, movable2waypoint).round(3) < 0:
                        print(f'{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] skip waypoint {i+1} because it is moving in opposite direction of the final target{bcolors.ENDC}')
                        continue
                # execute waypoint
                controller_info = lmp_env.ur5.execute(movable_var, waypoint)

                dist2target = np.linalg.norm(movable_var['_position_world'] - queue_list[-1][0])
                if not object_centric and controller_info['mp_info'] == -1:
                    print(f'{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] failed waypoint {i+1} (wp: {waypoint[0].round(3)}, actual: {movable_var["_position_world"].round(3)}, target: {queue_list[-1][0].round(3)}, start: {queue_list[0][0].round(3)}, dist2target: {dist2target.round(3)}); mp info: {controller_info["mp_info"]}{bcolors.ENDC}')
                else:
                    print(f'{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] completed waypoint {i+1} (wp: {waypoint[0].round(3)}, actual: {movable_var["_position_world"].round(3)}, target: {queue_list[-1][0].round(3)}, start: {queue_list[0][0].round(3)}, dist2target: {dist2target.round(3)}){bcolors.ENDC}')
                i += 1
            print(f'{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] finished executing path via controller{bcolors.ENDC}')

        if not object_centric:
            try:
                # traj_world: world_xyz, rotation, velocity, gripper
                ee_pos_world = queue_list[-1][0]
                ee_rot_world = queue_list[-1][1]
                ee_pose_world = np.concatenate([ee_pos_world, ee_rot_world])
                ee_speed = queue_list[-1][2]
                gripper_state = queue_list[-1][3]
            except:
                # evaluate latest voxel map
                _rotation_map = rotation_map
                _velocity_map = velocity_map
                _gripper_map = gripper_map
                # get last ee pose
                ee_pos_world = lmp_env._env.get_ee_pos()
                ee_pos_voxel = lmp_env.get_ee_pos()
                ee_rot_world = _rotation_map[ee_pos_voxel[0], ee_pos_voxel[1], ee_pos_voxel[2]]
                ee_pose_world = np.concatenate([ee_pos_world, ee_rot_world])
                ee_speed = _velocity_map[ee_pos_voxel[0], ee_pos_voxel[1], ee_pos_voxel[2]]
                gripper_state = _gripper_map[ee_pos_voxel[0], ee_pos_voxel[1], ee_pos_voxel[2]]
            # move to the final target
            lmp_env._env.apply_action(np.concatenate([ee_pose_world, [gripper_state]]))


    def __call__(self, query, file_lock, lmp_env):
        planning = self.generate_planning(query)
        planning_ = planning.copy()
        update_stop_event = threading.Event()
        exec_stop_event = threading.Event()
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
            print(f"Action: {action}")
            if action_state is None:
                filepath = self.get_last_filename(self.mask_path)
                action_state  = self._vlmapi_call(filepath, query=query, planner=planning_, action=action, objects=self._context)
            print(action_state)

            # 启动更新路径的线程
            update_thread = threading.Thread(target=self.__thread_update_traj, args=(lmp_env, action_state, file_lock, update_stop_event,exec_stop_event, ))
            update_thread.daemon = True  # 设置为守护线程，随主线程退出
            update_thread.start()

            # 启动执行路径的线程
            execute_thread = threading.Thread(target=self.__thread_execute_traj, args=(lmp_env, action_state, file_lock, update_stop_event,exec_stop_event, ))
            execute_thread.daemon = True  # 设置为守护线程，随主线程退出
            execute_thread.start()

            execute_thread.join()
            update_thread.join()

            update_stop_event.clear()
            exec_stop_event.clear()

            # Clear old queue and insert new trajectory
            while not self.shared_queue.empty():
                try:
                    self.shared_queue.get_nowait()
                except Exception:
                    break

            if len(planning) == 0:
                print(f"{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] finished all planning{bcolors.ENDC}")
                time.sleep(1)
                lmp_env.reset_to_default_pose()
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
