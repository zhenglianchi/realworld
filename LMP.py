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
from scipy.ndimage import distance_transform_edt,zoom
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


class AdaptiveWeightAdjuster:
    """自适应权重调节（Adaptive Weight Modulation），实现与论文公式一致。

    论文中局部动作选择代价为 p_{t+1} = argmin [ β·L_dir + α·T_t + γ·O_t ]，
    其中目标吸引权重 α 与避障权重 γ 随场景瞬时变化自适应调节（β 固定）：
        1. 障碍物权重 γ：机器人当前位置附近障碍物越密集，γ 越大
        2. 目标权重 α：机器人离目标越远，α 越大（加速靠近）
        3. 方向引导权重 β 保持固定不变

    具体公式（完全基于当前帧计算，无历史依赖，瞬时响应）：
        d_obs    = mean_{v ∈ N(p_t, r)} O_t(v)          # 局部障碍物密度
        d_target = ‖p_t − p_tar‖ / (map_size·√2)        # 归一化目标距离（体素系 xy 对角线）
        d⁺       = max(0, d − τ)                         # 截断超额量
        Δ(d⁺)    = σ(10·d⁺ − 5)                          # 平移 sigmoid 映射
        α        = w_target·(1 + Δ(d_target⁺))
        γ        = w_obs·(1 + Δ(d_obs⁺))
    不需要训练，所有调整基于当前统计信息。
    """
    def __init__(self, base_weights,
                 tau_obs=0.15, tau_target=0.2,
                 local_radius=5):
        """
        Args:
            base_weights: dict with keys {'obstacle': w0_obs, 'target': w0_target, 'direction': w0_dir}
                基础权重，来自配置文件
            tau_obs: 障碍物密度阈值 ∈ [0, 1]，低于此值几乎不增加权重，默认0.15
            tau_target: 目标距离阈值 ∈ [0, 1]，超过此值才显著增加权重，默认0.2
                         (归一化后，0.2 ≈ 地图对角线的20%)
            local_radius: 局部密度计算半径，默认5体素
        """
        # 基础权重，方向权重固定不调整
        self.w_obs_0 = base_weights['obstacle']
        self.w_target_0 = base_weights['target']
        self.w_dir = base_weights['direction']

        # 超参数
        self.tau_obs = tau_obs
        self.tau_target = tau_target
        self.local_radius = int(local_radius)

        # 不需要保存任何历史状态，完全基于当前帧计算，瞬时响应

    def _calc_local_density(self, current_pos, avoidance_map):
        """Calculate obstacle density in local neighborhood around current position."""
        H, W, D = avoidance_map.shape
        r = self.local_radius

        # Get neighborhood bounds with clipping
        x0 = max(0, int(np.round(current_pos[0])) - r)
        x1 = min(H, int(np.round(current_pos[0])) + r + 1)
        y0 = max(0, int(np.round(current_pos[1])) - r)
        y1 = min(W, int(np.round(current_pos[1])) + r + 1)
        z0 = max(0, int(np.round(current_pos[2])) - r)
        z1 = min(D, int(np.round(current_pos[2])) + r + 1)

        # Extract local patch and calculate density
        local_patch = avoidance_map[x0:x1, y0:y1, z0:z1]
        density = float(np.mean(local_patch))
        return density

    def _get_target_centroid(self, affordable_map):
        """Get centroid of target region from affordable_map."""
        coords = np.argwhere(affordable_map > 0)
        if len(coords) == 0:
            return np.zeros(3, dtype=np.float32)
        return np.mean(coords, axis=0).astype(np.float32)

    def _sigmoid_smooth(self, x):
        """Smooth sigmoid mapping that maps x ∈ [0,1] → Δ ∈ [0,1].

        Scaled sigmoid: Δ = σ(10x - 5)
        Weight = 10.0, bias = -5.0
        Property:
        - x ∈ [0, 1] → 10x-5 ∈ [-5, 5] → Δ ∈ [≈0.007, ≈0.993]
        - Symmetric around x=0.5: Δ(0.5) = 0.5
        - Smooth sigmoid, gradually increasing from near 0 to near 1
        - Strictly increasing, smooth saturation
        - Used for smooth parameter amplification.
        """
        # Weight 10.0, bias -5.0
        # Symmetric: x=0 → Δ≈0, x=0.5 → Δ=0.5, x=1 → Δ≈1
        return 1.0 / (1.0 + np.exp(-(10.0 * x - 5.0)))

    def update(self, current_pos, affordable_map, avoidance_map, map_size):
        """
        Update adaptive weights based on current map data.
        完全基于当前观测，瞬时响应，无任何历史依赖。

        Args:
            current_pos: (3,) current robot end-effector position in voxel coordinates
            affordable_map: (H, W, D) target/affordance distance map
            avoidance_map: (H, W, D) obstacle occupancy map
            map_size: size of voxel grid

        Returns:
            dict: {'obstacle': w_obs, 'target': w_target, 'direction': w_dir}
                自适应计算得到的权重
        """
        current_pos = np.array(current_pos, dtype=np.float32)

        # ========== 1. 障碍物权重自适应（论文：γ = w_obs·(1 + Δ(d_obs⁺))）==========
        # 局部障碍物密度越大 → 障碍物惩罚权重 γ 越大
        # density ∈ [0, 1]，量纲已经归一化，不需要额外处理
        density = self._calc_local_density(current_pos, avoidance_map)
        # d⁺ = max(0, d − τ_obs)，只在密度超过阈值时增大权重，否则保持基权重
        # max(0, ...) 保证只有正值调整，不减少权重，只放大
        x_obs_pos = max(0.0, density - self.tau_obs)  # x_obs_pos ∈ [0, 1-τ_obs] ⊂ [0, 1]
        # sigmoid 平滑映射：Δ = σ(10·d⁺ − 5)，饱和非线性
        # 只放大不缩小，Δ=0 时权重不变
        adjust_obs = self._sigmoid_smooth(x_obs_pos)
        w_obs = self.w_obs_0 * (1 + adjust_obs)

        # ========== 2. 目标权重自适应（论文：α = w_target·(1 + Δ(d_target⁺))）==========
        # 机器人离目标越远 → 目标吸引权重 α 越大，加速机器人靠近
        # 按地图 xy 平面对角线最大距离归一化到 [0, 1]（论文的 √(W²+H²)），与密度量纲一致
        # 地面操作只考虑 xy 平面，不包含 z 轴
        current_target = self._get_target_centroid(affordable_map)
        dist_to_target = np.linalg.norm(current_pos - current_target)
        # 归一化：除以地图 xy 对角线最大距离 → 结果 ∈ [0, 1]
        max_dist = map_size * np.sqrt(2)  # xy 平面对角线
        norm_dist = dist_to_target / max_dist
        # d⁺ = max(0, d − τ_target)，只在距离超过阈值时增大权重，否则保持基权重
        # max(0, ...) 保证只有正值调整，不减少权重，只放大
        x_target_pos = max(0.0, norm_dist - self.tau_target)  # ∈ [0, 1-τ_target] ⊂ [0, 1]
        # sigmoid 平滑映射：Δ = σ(10·d⁺ − 5)，饱和非线性
        adjust_target = self._sigmoid_smooth(x_target_pos)
        w_target = self.w_target_0 * (1 + adjust_target)

        # ========== 3. 方向权重固定（β 不参与自适应，保持全局对齐）==========
        w_dir = self.w_dir

        return {
            'obstacle': w_obs,
            'target': w_target,
            'direction': w_dir
        }

    def reset(self):
        """Reset states. Call when starting a new planning task.
        因为不保存历史，所以不需要重置任何东西。
        """
        pass


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

        # 自适应权重调节器（论文：α/γ 随目标距离与障碍密度动态调节，β 固定）
        self.adaptive_weight_adjuster = None

    def get_last_filename(self,folder):
        while True:
            filenames = os.listdir(folder)
            if len(filenames) != 0:
                filename = filenames[-1]
                return f"{folder}{filename}"
            else:
                time.sleep(1)

    def get_response(self, messages, caller="unknown"):
        client = OpenAI(api_key=self.api_key, base_url=self.base_url)

        _vlm_start = time.time()
        completion = client.chat.completions.create(
            model=self._cfg['vision_model'],
            messages=messages
        )
        _vlm_elapsed = time.time() - _vlm_start
        print(f"[LOG][VLM推理] VLM API调用延时: {_vlm_elapsed:.3f}s, 调用者: {caller}, 模型: {self._cfg['vision_model']}")
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

        result = self.get_response(messages, caller="generate_planning")

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
            tokens = self.get_response(messages, caller="rotation_generate").choices[0].message.content
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
                #pause = input("press any key to continue...")
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
            #lmp_env.ur5.servoL(next_pos,time=4)
            #time.sleep(8)

            


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

        result = self.get_response(messages, caller="_vlmapi_call")

        resstr = result.choices[0].message.content.replace("```","").replace("json","")

        print(resstr)

        state = json.loads(resstr)
        if isinstance(state, list):
            return state[0]
        return state



    def __get__affordable_map(self,action_state,lmp_env,object_state):
        affordable_map = lmp_env._get_default_voxel_map('target')()
        affordable = action_state["affordable"]
        affordable_set = affordable["set"]
        if affordable_set != "default" :
            affordable_var = affordable["object"]
            if affordable_var not in object_state:
                print(f"Object {affordable_var} not found in scene in this step, using default affordable map.")
                return affordable_map
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
            if not isinstance(avoidance_vars, list):
                avoidance_vars = [avoidance_vars]

            for avoidance_var in avoidance_vars:
                if avoidance_var not in object_state.keys():
                    print(f"Object {avoidance_var} not found in scene in this step.")
                    pass
                object = object_state[avoidance_var]["obs"]
                if "occupancy_map" in avoidance.keys():
                    occupancy_map = eval(avoidance["occupancy_map"])
                value = avoidance["value"]
                avoidance_map = set_voxel_by_avoid(avoidance_map, occupancy_map, value)
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
            if gripper_var not in object_state:
                print(f"Object {gripper_var} not found in scene in this step, using default gripper map.")
                return gripper_map
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
    
    def get_cost_map(self,lmp_env, affordable_map, avoidance_map, current_pos=None):
        target_map = affordable_map
        #obstacle_map = gaussian_filter(avoidance_map, sigma=lmp_env.slow_planner.config.obstacle_map_gaussian_sigma)
        #obstacle_map = normalize_map(obstacle_map)
        obstacle_map = avoidance_map

        # 论文 eq. L_t(u) = T_t(u) + O_t(u)：慢系统规划代价为目标吸引场与
        # 障碍代价场（单位代价）之和，两者均不参与自适应加权。
        costmap = target_map + obstacle_map

        # 自适应权重（论文 eq. α = w_target(1+Δ(d_target⁺))、γ = w_obs(1+Δ(d_obs⁺))）
        # 作用于快系统局部动作选择的代价系数：β 固定，α / γ 动态调节。
        if (self.adaptive_weight_adjuster is not None and
            hasattr(lmp_env.slow_planner.config, 'adaptive_weights_enabled') and
            lmp_env.slow_planner.config.adaptive_weights_enabled and
            current_pos is not None):
            # Get adaptive weights
            map_size = lmp_env.slow_planner.map_size
            adaptive_weights = self.adaptive_weight_adjuster.update(
                current_pos, affordable_map, avoidance_map, map_size
            )
            # Update fast planner adaptive coefficients (α: target, γ: obstacle, β: direction)
            if hasattr(lmp_env.fast_planner, 'adaptive_alpha'):
                lmp_env.fast_planner.adaptive_alpha = adaptive_weights['target']
                lmp_env.fast_planner.adaptive_avoid_weight = adaptive_weights['obstacle']
                lmp_env.fast_planner.adaptive_beta = adaptive_weights['direction']
        else:
            # 非自适应模式：快系统回落到基线权重（与慢系统相同的 w_target / w_obs 基线）
            if hasattr(lmp_env.fast_planner, 'adaptive_alpha'):
                lmp_env.fast_planner.adaptive_alpha = lmp_env.slow_planner.config.target_map_weight
                lmp_env.fast_planner.adaptive_avoid_weight = lmp_env.slow_planner.config.obstacle_map_weight
                lmp_env.fast_planner.adaptive_beta = None  # β 使用 fast_planner 自身基线（固定）

        costmap = normalize_map(costmap)
        return costmap

    def distance_map_from_single_target(self, target_map, normalize=True):
        """
        从单目标点二值图生成欧氏距离场。
        
        Args:
            target_map: 3D array, shape (H, W, D), exactly one voxel == 1 (or >0), rest == 0.
            normalize: bool, whether to normalize to [0, 1].
        
        Returns:
            distance_map: float32 array, same shape as input.
        """
        coords = np.argwhere(target_map)
        if coords.size == 0:
            raise ValueError("No target point found!")
        target_point = coords[0]
        
        H, W, D = target_map.shape
        x, y, z = target_point

        xx = np.arange(H, dtype=np.float32) - x
        yy = np.arange(W, dtype=np.float32) - y
        zz = np.arange(D, dtype=np.float32) - z

        dist = np.sqrt(xx[:, None, None]**2 + yy[None, :, None]**2 + zz[None, None, :]**2)
        
        if normalize:
            dist /= dist.max()

        return dist


    def __thread_update_map(self, lmp_env, action_state, state_manager):
        global _map_size, _resolution
        _map_size = lmp_env._map_size
        _resolution = lmp_env._resolution
        update_count = 0
        total_latency = 0.0
        last_log_time = time.time()

        while not self.update_stop_event.is_set():
            test1 = time.time()
            object_state = state_manager.get_state(blocking = True, timeout = 300.0)
            affordable_map = self.__get__affordable_map(action_state,lmp_env,object_state)
            if not self.wakeup_flag:
                gripper_map = self.__get__gripper_map(action_state,lmp_env,object_state)

            avoidance_map = self.__get__avoidance_map(action_state,lmp_env,object_state)

            affordable_map = self.distance_map_from_single_target(affordable_map)

            movable = action_state["movable"]
            movable_var = object_state[movable]["obs"]
            test2 = time.time()

            with self.map_lock:
                self.movable_var = movable_var
                self.affordable_map = affordable_map
                self.avoidance_map = avoidance_map
                self.gripper_map = gripper_map
                update_count += 1
                total_latency += test2 - test1

            if not self.wakeup_flag:
                with self.init_condition:
                    self.init_condition.notify()
                    self.wakeup_flag = True
            # 每2秒打印一次更新次数和平均延时
            current_time = time.time()
            if current_time - last_log_time >= 2.0:
                avg_lat = (total_latency / update_count * 1000) if update_count > 0 else 0
                fps = update_count / (current_time - last_log_time)
                print(f"{bcolors.OKGREEN}[Target Map] {update_count}次更新, "
                      f"平均延时{avg_lat:.1f}ms, 频率{fps:.1f}Hz{bcolors.ENDC}")
                update_count = 0
                total_latency = 0.0
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

            costmap = self.get_cost_map(lmp_env, affordance_map, avoidance_map, start_pos)
            
            # Optimize path and log
            lmp_env.slow_planner.optimize(start_pos, costmap, self.shared_queue)
            assert not self.shared_queue.empty(), 'path_voxel is empty'

            end_time = time.time()
            print(f"{bcolors.OKBLUE}[Slow Planner] 贪婪规划延时{(end_time - start_time)*1000:.1f}ms{bcolors.ENDC}")



    def __thread_execute_traj(self, lmp_env):
        num = 0
        while not self.exec_stop_event.is_set():
            movable_var, affordable_map, avoidance_map, gripper_map = self.get_map()
            num += 1
            if num <= 5:
                time_sleep = 0.7
            else:
                time_sleep = 0.45

            if self.shared_queue.size() == 0:
                if self.update_stop_event.is_set():
                    curr_xyz = lmp_env.ur5.get_tcp()[:3]  # 直接获取实时位置
                    rotation = lmp_env.ur5.get_tcp()[3:]
                    current_voxel_xyz = np.array(lmp_env._world_to_voxel(curr_xyz))
                    
                    gripper = gripper_map[current_voxel_xyz[0], current_voxel_xyz[1], current_voxel_xyz[2]]

                    waypoint = (curr_xyz, rotation, gripper)
                    # execute waypoint
                    #lmp_env.ur5.execute(waypoint, time_sleep)
                    #self.exec_stop_event.set()
                    break
                else:
                    continue

            curr_xyz = lmp_env.ur5.get_tcp()[:3]  # 直接获取实时位置
            rotation = lmp_env.ur5.get_tcp()[3:]
            current_voxel_xyz = np.array(lmp_env._world_to_voxel(curr_xyz))

            _fast_start = time.time()
            voxel_xyz,queue_list = lmp_env.fast_planner.generate_fast_point_3d_vectorized(current_voxel_xyz, self.shared_queue, affordable_map, avoidance_map)
            _fast_elapsed = time.time() - _fast_start
            world_xyz = lmp_env._voxel_to_world(voxel_xyz)
            voxel_xyz = np.round(voxel_xyz).astype(int)
            
            gripper = gripper_map[voxel_xyz[0], voxel_xyz[1], voxel_xyz[2]]
        
            waypoint = (world_xyz, rotation, gripper)
                
            # execute waypoint
            #signal = lmp_env.ur5.execute(waypoint, time_sleep)

            '''if not signal :
                print(f"{bcolors.FAIL}[interfaces.py | {get_clock_time()}] Failed to execute waypoint{bcolors.ENDC}")
                self.exec_stop_event.set()
                self.update_stop_event.set()
                break'''

            self.executed_path_voxel.append(voxel_xyz.copy())
            time.sleep(time_sleep)

            if len(queue_list) == 0:
                pass
            else:
                final_target = lmp_env._voxel_to_world(queue_list[-1])
            dist2target = np.linalg.norm(curr_xyz - final_target)

            print(f'{bcolors.OKBLUE}[Fast Planner] 选点延时{_fast_elapsed*1000:.1f}ms, '
                  f'wp:{waypoint[0].round(3)}, dist2target:{dist2target.round(3)}{bcolors.ENDC}')

            # check if the movement is finished 1.5cm
            if dist2target <= 0.015 or self.shared_queue.size() <= 1:
                print(f"{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] reached last waypoint; curr_xyz={curr_xyz}, target={final_target} (distance: {dist2target:.3f})){bcolors.ENDC}")
                self.exec_stop_event.set()
                self.update_stop_event.set()
                waypoint = (final_target, rotation, gripper)
                #lmp_env.ur5.execute(waypoint, time_sleep)
                #lmp_env.ur5.gripper.gripper_close()
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
                if moving:
                    time.sleep(5)
                    continue
                lmp_env.ur5.reset_to_default_pose()
                time.sleep(5)
                continue
            if action == "close the gripper" or action == "close gripper":
                if hasattr(lmp_env.ur5, 'gripper') and lmp_env.ur5.gripper is not None:
                    lmp_env.ur5.gripper.gripper_close()
                else:
                    print("Gripper not connected, skipping gripper close.")
                time.sleep(5)
                continue
            if action == "open the gripper" or action == "open gripper":
                if hasattr(lmp_env.ur5, 'gripper') and lmp_env.ur5.gripper is not None:
                    lmp_env.ur5.gripper.gripper_open()
                else:
                    print("Gripper not connected, skipping gripper open.")
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

            # Initialize/reset adaptive weight adjuster
            if (hasattr(lmp_env.slow_planner.config, 'adaptive_weights_enabled') and
                lmp_env.slow_planner.config.adaptive_weights_enabled and
                self.adaptive_weight_adjuster is None):
                # Get base weights from configuration
                base_weights = {
                    'obstacle': lmp_env.slow_planner.config.obstacle_map_weight,
                    'target': lmp_env.slow_planner.config.target_map_weight,
                    'direction': lmp_env.fast_planner.beta
                }
                # Create adaptive weight adjuster (论文公式：α = w_target(1+Δ), γ = w_obs(1+Δ))
                from LMP import AdaptiveWeightAdjuster
                self.adaptive_weight_adjuster = AdaptiveWeightAdjuster(
                    base_weights,
                    tau_obs=lmp_env.slow_planner.config.adaptive_tau_obs,
                    tau_target=lmp_env.slow_planner.config.adaptive_tau_target,
                    local_radius=lmp_env.slow_planner.config.adaptive_local_radius
                )
            # Reset for new task
            if self.adaptive_weight_adjuster is not None:
                self.adaptive_weight_adjuster.reset()

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

                # 生成文件名
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                action_name = action_state["Action"].replace(" ", "_")
                filename = f"{timestamp}_{action_name}.html"

                # 可视化
                voxel_visualizer.visualize(
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

def set_voxel_by_avoid(voxel_map, points_vox, value):
  # 1. 去重原始点
    radius_vox = 4
    # 去重并转为整数
    points = np.unique(np.asarray(points_vox, dtype=int), axis=0)

    H, W, D = voxel_map.shape  # 支持非立方体地图

    for pt in points:
        x, y, z = pt
        
        min_x = max(0, x - radius_vox)
        max_x = min(H, x + radius_vox + 1)
        min_y = max(0, y - radius_vox)
        max_y = min(W, y + radius_vox + 1)
        min_z = max(0, z - radius_vox)
        max_z = min(D, z + radius_vox + 1)

        # 切片赋值（高效！）
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