from openai import OpenAI
from utils import load_prompt,normalize_vector,bcolors,get_clock_time
from VLM_demo import encode_image
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
from scipy.ndimage import gaussian_filter
from utils import get_clock_time, normalize_map

class SharedQueue:
    """
    线程安全的共享队列，支持：
    - 写入：传入一个 array 列表，将其元素按顺序放入队列
    - 读取：实时读取当前队列内容（返回 list），无更新时返回旧队列
    """
    def __init__(self):
        self._queue = queue.Queue()
        self._data = []  # 实时可读的列表（保护副本）
        self._lock = threading.Lock()  # 保证线程安全

    def put_all(self, items):
        """
        写入函数：传入一个 array 列表，将其元素按顺序放入队列
        并立即更新实时读取的副本
        """
        with self._lock:
            # 清空旧队列（可选：若要完全替换）
            while not self._queue.empty():
                try:
                    self._queue.get()
                except queue.Empty:
                    print("Queue is empty")
                    break

            # 批量写入新元素
            for item in items:
                self._queue.put(item)

            # 更新实时读取的副本
            self._data = list(self._queue.queue)

    def get_all(self):
        """
        实时读取函数：返回当前队列的副本（list）
        如果队列未更新，返回上一次的内容；更新后自动返回新内容
        """
        with self._lock:
            return self._data
        
    def remove_front(self):
        """
        外部调用：删除 _data 的第一个元素
        用于表示“该路径点已到达，不再需要”
        """
        with self._lock:
            if len(self._data) > 0:
                self._data.pop(0)

    def clear(self):
        """清空队列"""
        with self._lock:
            while not self._queue.empty():
                try:
                    self._queue.get()
                except queue.Empty:
                    print("Queue is empty")
                    break

            self._data.clear()

    def empty(self):
        """检查队列是否为空"""
        with self._lock:
            return self._queue.empty()

    def size(self):
        """获取队列大小"""
        with self._lock:
            return len(self._data)


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
        self._rotation_prompt = load_prompt(f"{env}/{self._cfg['rotation_prompt_fname']}.txt")

        self._stop_tokens = [self._cfg['stop']]
        self._context = None
        self.mask_path = "./tmp/masks/"
        self.image_path = "./tmp/images/"
        #set your api_key Qwen
        self.api_key= "sk-df55df287b2c420285feb77137467576"
        self.base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"

        self.update_stop_event = threading.Event()
        self.exec_stop_event = threading.Event()
        self.map_lock = threading.Lock()

        self.shared_queue = SharedQueue()
        self.executed_path_voxel = []
        self.init_condition = threading.Condition()
        self.wakeup_flag = False

        self.movable_var = None
        self.affordable_map = None
        self.avoidance_map = None
        self.gripper_map = None

        self.move = False

    def get_last_filename(self,folder):
        while True:
            filenames = os.listdir(folder)
            if len(filenames) != 0:
                filename = filenames[-1]
                return f"{folder}{filename}"
            else:
                time.sleep(1)

    def get_response(self,messages):
        client = OpenAI(api_key=self.api_key,base_url=self.base_url)

        completion = client.chat.completions.create(
            model=self._cfg['vision_model'], 
            messages=messages
        )
        return completion

    def generate_planning(self, query, lmp_env,image_share):
        self._context = lmp_env.objects

        user_query = f'{self._cfg["query_prefix"]}{query}{self._cfg["query_suffix"]}'

        planner_prompt = self._planner_prompt

        if self._context :
            user_query = f"# Objects : {self._context}\n" + user_query

        print(user_query)
        
        base64_image = image_share.get()

        messages=[{"role": "user","content": [
                {"type": "text","text": f"This is a robotic arm operation scene image.\n{planner_prompt}\nThe above are some examples of planning, please give the corresponding planning according to the image I gave you next:\n{user_query}. The output format likely is\n" + "planner : ['', '', '', '']\nOther than that, don't give me any superfluous information and hints.The objects in the generated plan should match the names in the given image"},
                {"type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}, 
                }
                ]}]

        result = self.get_response(messages)

        planner = result.choices[0].message.content

        planning = json.loads(planner.split(":")[-1].strip())

        return planning
    
    def rotation_generate(self, action,lmp_env, image_share):
        base64_image = image_share.get()
        axis = ["z","x","y"]

        messages = []

        Q=[
            "What is the orientation of the gripper's opening when performing this motion?Please determine whether rotation is necessary based on the representation of the gripper in the image and the coordinate system.\n(A) Vertical (B) Horizontal (C) Default",
            "Can the current gripper opening direction performing this motion? If needed, rotate how many degrees around the Z-axis?Please determine whether rotation is necessary based on the representation of the gripper in the image and the coordinate system.\n(A) Yes (B) No (degrees)__",
            "Can the current gripper opening direction performing this motion? If needed, rotate how many degrees around the X-axis?Please determine whether rotation is necessary based on the representation of the gripper in the image and the coordinate system.\n(A) Yes (B) No (degrees)__",
            "Can the current gripper opening direction performing this motion? If needed, rotate how many degrees around the Y-axis?Please determine whether rotation is necessary based on the representation of the gripper in the image and the coordinate system.\n(A) Yes (B) No (degrees)__",
        ]

        prompt = f"{self._rotation_prompt}"
        messages.append({"role": "user","content": [
            {"type": "text",
             "text": f"This is a robotic arm operation scene." + f"The format of output should be like {prompt}.\nAction: {action}\n"},
            {"type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}, 
            }
            ]})
        messages.append({"role": "assistant","content": [{"type": "text","text": "Understood, let's start the question and answer sequence."}]})
            
        
        for i in range(len(Q)):
            current_pos = lmp_env.ur5.get_tcp()[:3]
            current_rotation = lmp_env.ur5.get_tcp()[3:]
            base64_image = image_share.get()
            messages.append({
                "role": "user","content": [
                    {"type": "text","text": Q[i]},
                    {"type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}, 
                    }
                    ]
                             }
                             )
            tokens = self.get_response(messages).choices[0].message.content
            answer = tokens.split("\n")[1].strip()
            explanation = tokens.split("\n")[2]
            print(Q[i])
            print(tokens)
            messages.append({"role": "assistant","content": [{"type": "text","text": tokens}]})

            C = answer[1]

            if i>0:
                A = int(answer[3:])
                if A>0:
                    pause = input("press any key to continue...")

                if C == "A":
                    if i == 2:
                        target_rotation = rotate_pose_local_axis(target_rotation,axis[i-1],A)
                    elif i == 3 or i == 1:
                        target_rotation = rotate_pose_local_axis(target_rotation,axis[i-1],-A)
                elif C == "B":
                    target_rotation = current_rotation
                else:
                    print("Invalid choice")
                    exit()
            else:
                pause = input("press any key to continue...")
                if C == "A" or C == "C":
                    # 如果是A则恢复到初始位姿
                    target_rotation = current_rotation
                elif C == "B":
                    target_rotation = rotate_pose_local_axis(current_rotation,'y',-90)
                else:
                    print("Invalid choice")
                    exit()


            next_pos = np.array(current_pos.tolist()+target_rotation.tolist())
            #print(next_pos)
            lmp_env.ur5.servoL(next_pos,time=4)
            time.sleep(8)

            


    def _vlmapi_call(self, image_share,query, planner ,action, objects):

        base64_image = image_share.get()

        prompt = self._action_state_prompt

        print(objects)

        messages=[{"role": "user","content": [
                {"type": "text","text": f"This is a robotic arm operation scene." + f"The format of output should be like {prompt}.\n Objects : {objects}\nMoves : [grasp,move],\nQuery : {query}\nPlanner : {planner}\nAction : {action}\nPlease just give me the corresponding json, no explanation and no text required"},
                {"type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}, 
                }
                ]}]

        result = self.get_response(messages)

        resstr = result.choices[0].message.content.replace("```","").replace("json","")

        print(resstr)

        state = json.loads(resstr)

        return state[0]



    def __get__affordable_map(self,action_state,lmp_env,object_state):
        affordable_map = lmp_env._get_default_voxel_map('target')()
        affordable = action_state["affordable"]
        affordable_set = affordable["set"]
        if affordable_set != "default" :
            affordable_map = lmp_env._get_default_voxel_map('target')()
            affordable_var = affordable["object"]
            object = object_state[affordable_var]["obs"]
            if "center_x, center_y, center_z" in affordable.keys():
                center_x, center_y, center_z = eval(affordable["center_x, center_y, center_z"])
            if "(min_x, min_y, min_z), (max_x, max_y, max_z)" in affordable.keys():
                (min_x, min_y, min_z), (max_x, max_y, max_z) = eval(affordable["(min_x, min_y, min_z), (max_x, max_y, max_z)"])
            x = eval(affordable["x"])
            y = eval(affordable["y"])
            z = eval(affordable["z"])
            target_affordance = affordable["target_affordance"]
            x,y,z = np.array([x,y,z]).astype(int)
            affordable_map[x,y,z] = target_affordance
            self.move = True
            print(x,y,z)
        return affordable_map
    
    def __get__avoidance_map(self,action_state,lmp_env,object_state):
        avoidance_map = lmp_env._get_default_voxel_map('obstacle')()
        avoidance = action_state["avoid"]
        avoidance_set = avoidance["set"]
        if avoidance_set != "default" :
            avoidance_vars = action_state["avoid"]["object"]
            avoidance_vars = eval(avoidance_vars)
            if not isinstance(avoidance_vars, list):
                avoidance_vars = [avoidance_vars]

            for avoidance_var in avoidance_vars:
                if avoidance_var not in object_state.keys():
                    print(f"Object {avoidance_var} not found in scene in this step.")
                    pass
                object = object_state[avoidance_var]["obs"]
                if "center_x, center_y, center_z" in avoidance.keys():
                    center_x, center_y, center_z = eval(avoidance["center_x, center_y, center_z"])
                if "(min_x, min_y, min_z), (max_x, max_y, max_z)" in avoidance.keys():
                    (min_x, min_y, min_z), (max_x, max_y, max_z) = eval(avoidance["(min_x, min_y, min_z), (max_x, max_y, max_z)"])
                x = eval(avoidance["x"])
                y = eval(avoidance["y"])
                z = eval(avoidance["z"])
                radius_cm = avoidance["radius_cm"]
                value = avoidance["value"]
                avoidance_map = set_voxel_by_radius(avoidance_map, [x,y,z], radius_cm, value)
        return np.array(avoidance_map)
    
    def __get__gripper_map(self,action_state,lmp_env,object_state):
        gripper_map = lmp_env._get_default_voxel_map('gripper')()
        gripper = action_state["gripper"]
        gripper_set = gripper["set"]
        if gripper_set != "default" :
            if gripper_set == "gripper_map[:, :, :] = 1" :
                gripper_map[:, :, :] = 1
                return gripper_map
            elif gripper_set == "gripper_map[:, :, :] = 0" :
                gripper_map[:, :, :] = 0
                return gripper_map
            gripper_var = action_state["gripper"]["object"]
            object = object_state[gripper_var]["obs"]
            if "center_x, center_y, center_z" in gripper.keys():
                center_x, center_y, center_z = eval(gripper["center_x, center_y, center_z"])
            if "(min_x, min_y, min_z), (max_x, max_y, max_z)" in gripper.keys():
                (min_x, min_y, min_z), (max_x, max_y, max_z) = eval(gripper["(min_x, min_y, min_z), (max_x, max_y, max_z)"])
            x = eval(gripper["x"])
            y = eval(gripper["y"])
            z = eval(gripper["z"])
            radius_cm = gripper["radius_cm"]
            value = gripper["value"]
            gripper_map = set_voxel_by_radius(gripper_map, [x,y,z], radius_cm, value)
        return gripper_map
    

    def get_map(self):
        with self.map_lock:
            return self.movable_var, self.affordable_map, self.avoidance_map, self.gripper_map
    
    def get_cost_map(self,lmp_env, affordable_map, avoidance_map):
        target_map = affordable_map
        obstacle_map = gaussian_filter(avoidance_map, sigma=lmp_env.slow_planner.config.obstacle_map_gaussian_sigma)
        obstacle_map = normalize_map(obstacle_map)
        # combine target_map and obstacle_map
        costmap = target_map * lmp_env.slow_planner.config.target_map_weight + obstacle_map * lmp_env.slow_planner.config.obstacle_map_weight
        costmap = normalize_map(costmap)
        return costmap


    def __thread_update_map(self, lmp_env, action_state, state_manager):
        global _map_size, _resolution
        _map_size = lmp_env._map_size
        _resolution = lmp_env._resolution
        update_count = 0
        last_log_time = time.time()

        while not self.update_stop_event.is_set():
            test1 = time.time()
            object_state = state_manager.get_state(blocking = True, timeout = 300.0)
            #print(object_state)
            affordable_map = self.__get__affordable_map(action_state,lmp_env,object_state)
            if not self.wakeup_flag:
                gripper_map = self.__get__gripper_map(action_state,lmp_env,object_state)

            avoidance_map = self.__get__avoidance_map(action_state,lmp_env,object_state)
            
            affordable_map = distance_transform_edt(1 - affordable_map)
            affordable_map = normalize_map(affordable_map)

            movable = action_state["movable"]
            movable_var = object_state[movable]["obs"]
            test2 = time.time()
            print("test:",test2-test1)

            with self.map_lock:
                self.movable_var = movable_var
                self.affordable_map = affordable_map
                self.avoidance_map = avoidance_map
                self.gripper_map = gripper_map
                update_count += 1

            if not self.wakeup_flag:
                with self.init_condition:
                    self.init_condition.notify()
                    self.wakeup_flag = True
            # 每2秒打印一次更新次数
            current_time = time.time()
            if current_time - last_log_time >= 2.0:
                fps = update_count / (current_time - last_log_time)
                print(f"{bcolors.OKGREEN}[interfaces.py | {get_clock_time()}] Map update rate: {update_count} updates in {current_time - last_log_time:.2f}s ({fps:.1f} Hz){bcolors.ENDC}")
                update_count = 0
                last_log_time = current_time


            
    def __thread_update_traj(self, lmp_env):
        while not self.update_stop_event.is_set():
            if not self.move:
                print("Gripper manipulation, no need to update traj")
                self.update_stop_event.set()
                break

            start_time = time.time()

            movable_var, affordance_map, avoidance_map, gripper_map = self.get_map()

            start_pos = lmp_env.get_ee_pos().copy()  # 直接获取实时位置
            
            costmap = self.get_cost_map(lmp_env, affordance_map, avoidance_map)
            
            # Optimize path and log
            lmp_env.slow_planner.optimize(start_pos, costmap, self.shared_queue)
            assert not self.shared_queue.empty(), 'path_voxel is empty'

            end_time = time.time()
            print(f"{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] updated trajectory in {end_time - start_time:.3f}s{bcolors.ENDC}")



    def __thread_execute_traj(self, lmp_env):
        num = 0
        while not self.exec_stop_event.is_set():
            movable_var, affordable_map, avoidance_map, gripper_map = self.get_map()
            num += 1
            if num <= 5:
                time_sleep = 0.7
            else:
                time_sleep = 0.5

            if self.shared_queue.size() == 0:
                if self.update_stop_event.is_set():
                    curr_xyz = lmp_env.ur5.get_tcp()[:3]  # 直接获取实时位置
                    rotation = lmp_env.ur5.get_tcp()[3:]
                    current_voxel_xyz = np.array(lmp_env._world_to_voxel(curr_xyz))
                    
                    gripper = gripper_map[current_voxel_xyz[0], current_voxel_xyz[1], current_voxel_xyz[2]]

                    waypoint = (curr_xyz, rotation, gripper)
                    # execute waypoint
                    lmp_env.ur5.execute(waypoint, time_sleep)
                    self.exec_stop_event.set()
                    break
                else:
                    continue

            curr_xyz = lmp_env.ur5.get_tcp()[:3]  # 直接获取实时位置
            rotation = lmp_env.ur5.get_tcp()[3:]
            current_voxel_xyz = np.array(lmp_env._world_to_voxel(curr_xyz))

            voxel_xyz,queue_list = lmp_env.fast_planner.generate_fast_point_3d_vectorized(current_voxel_xyz, self.shared_queue, affordable_map, avoidance_map)
            world_xyz = lmp_env._voxel_to_world(voxel_xyz)
            voxel_xyz = np.round(voxel_xyz).astype(int)
            
            gripper = gripper_map[voxel_xyz[0], voxel_xyz[1], voxel_xyz[2]]
        
            waypoint = (world_xyz, rotation, gripper)
                
            # execute waypoint
            signal = lmp_env.ur5.execute(waypoint, time_sleep)

            if not signal :
                print(f"{bcolors.FAIL}[interfaces.py | {get_clock_time()}] Failed to execute waypoint{bcolors.ENDC}")
                self.exec_stop_event.set()
                self.update_stop_event.set()
                break

            self.executed_path_voxel.append(voxel_xyz.copy())
            time.sleep(time_sleep)

            dist2target = np.linalg.norm(curr_xyz - lmp_env._voxel_to_world(queue_list[-1]))

            print(f'{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] completed waypoint: (wp: {waypoint[0].round(3)}, voxel: {voxel_xyz.round(3)}, actual: {movable_var["_position_world"].round(3)}, target: {queue_list[-1].round(3)}, start: {current_voxel_xyz}, dist2target: {dist2target.round(3)}){bcolors.ENDC}')

            # check if the movement is finished 5mm
            if dist2target <= 0.01 or self.shared_queue.size() == 0:
                print(f"{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] reached last waypoint; curr_xyz={curr_xyz}, target={queue_list[-1]} (distance: {dist2target:.3f})){bcolors.ENDC}")
                self.exec_stop_event.set()
                self.update_stop_event.set()
                break


    def __call__(self, query, lmp_env, state_manager, voxel_visualizer, image_share, moving):
        planning = self.generate_planning(query,lmp_env,image_share)
        planning = list(filter(None, planning))
        print(planning)
        planning_ = planning.copy()
        while len(planning) >= 0:
            self.wakeup_flag = False
            self.move = False

            if len(planning) == 0:
                print(f"{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] finished all planning{bcolors.ENDC}")
                break

            action = planning.pop(0)
            if action == "reset to default pose":
                lmp_env.ur5.reset_to_default_pose()
                time.sleep(5)
                continue
            if action == "close the gripper" or action == "close gripper":
                lmp_env.ur5.gripper.gripper_close()
                time.sleep(5)
                continue
            if action == "open the gripper" or action == "open gripper":
                lmp_env.ur5.gripper.gripper_open()
                time.sleep(5)
                continue

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
                action_state  = self._vlmapi_call(image_share, query=query, planner=planning_, action=action, objects=self._context)
                current_action = action_state["Action"]
                with open(f"./cache/{current_action}.json", 'w', encoding='utf-8') as json_file:
                    json.dump(action_state, json_file)

            print(action_state)

            if not moving:
                self.rotation_generate(action,lmp_env,image_share)
                pause = input("press any key to continue...")

            # 启动更新路径的线程
            map_thread = map_Thread(target=self.__thread_update_map, args=(lmp_env, action_state, state_manager))
            map_thread.daemon = True  # 设置为守护线程，随主线程退出

            # 启动更新路径的线程
            traj_thread = traj_Thread(target=self.__thread_update_traj, args=(lmp_env, ))
            traj_thread.daemon = True  # 设置为守护线程，随主线程退出

            # 启动执行路径的线程
            execute_thread = Low_Execute_Thread(target=self.__thread_execute_traj, args=(lmp_env, ))
            execute_thread.daemon = True  # 设置为守护线程，随主线程退出
            
            map_thread.start()
            with self.init_condition:
                self.init_condition.wait()
            traj_thread.start()
            execute_thread.start()

            execute_thread.join()
            traj_thread.join()
            map_thread.join()

            self.update_stop_event.clear()
            self.exec_stop_event.clear()
            
            self.shared_queue.clear()

            if self.move:
                print("正在生成可视化文件...")
                movable_var, affordable_map, avoidance_map, gripper_map = self.get_map()
                costmap = self.get_cost_map(lmp_env, affordable_map, avoidance_map)
                scenemap = lmp_env._get_scene_collision_voxel_map()

                # 生成文件名
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                action_name = action_state["Action"].replace(" ", "_")
                filename = f"{timestamp}_{action_name}.html"

                # 可视化
                voxel_visualizer.visualize(
                    scenemap=scenemap,
                    costmap=costmap,
                    executed_path_voxel=self.executed_path_voxel,
                    filename=filename,
                    show_cost_text=True
                )
                print("生成完毕...")

                # 清空
            self.executed_path_voxel.clear()



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

def rotate_pose_local_axis(rotvec_current, axis='x', angle_deg=30, degrees=True):
    """
    在当前姿态基础上，绕自身的 X/Y/Z 轴（局部坐标系）进行旋转，返回新的轴角表示。

    参数：
        rotvec_current (list or np.ndarray): 当前旋转向量 [Rx, Ry, Rz]（弧度）
        axis (str): 要绕的轴，支持 'x', 'y', 'z'（不区分大小写）
        angle_deg (float): 旋转角度（默认为度数）
        degrees (bool): 如果 True，angle_deg 是度数；否则是弧度

    返回：
        rotvec_new (np.ndarray): 旋转后的新旋转向量 [Rx, Ry, Rz]（弧度）
    """
    # 确保输入是 numpy 数组
    rotvec_current = np.array(rotvec_current).astype(float)
    
    # 角度转换：输入 angle_deg 是度数或弧度
    angle_rad = np.radians(angle_deg) if degrees else angle_deg

    # 1. 将当前旋转向量转为 Rotation 对象
    current_rotation = R.from_rotvec(rotvec_current)

    # 2. 创建绕指定局部轴的增量旋转（注意：绕自身轴 → 右乘，使用 *）
    if axis.lower() == 'x':
        delta_rotation = R.from_rotvec([angle_rad, 0, 0])
    elif axis.lower() == 'y':
        delta_rotation = R.from_rotvec([0, angle_rad, 0])
    elif axis.lower() == 'z':
        delta_rotation = R.from_rotvec([0, 0, angle_rad])
    else:
        raise ValueError("axis must be one of 'x', 'y', 'z'")

    # 3. 组合旋转：绕自身轴 = 当前姿态 × 增量旋转（右乘）
    # 即：新姿态 = 原姿态 + 在自身坐标系下的旋转
    new_rotation = current_rotation * delta_rotation

    # 4. 转回旋转向量
    rotvec_new = new_rotation.as_rotvec()  # 默认返回一维数组

    return rotvec_new