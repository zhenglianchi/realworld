from LMP import LMP
from utils import get_clock_time, normalize_vector, bcolors, Observation, VoxelIndexingWrapper
import numpy as np
from slow_planners import Slow_PathPlanner
from fast_planners import Fast_PathPlanner
import time
from scipy.ndimage import distance_transform_edt
import open3d as o3d
from VLM_demo import  write_state, get_world_bboxs_list,show_mask,process_visual_prompt,set_visual_prompt,predict_mask,encode_image,resize_bbox_to_original,smart_resize,get_response
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
import json
import json_numpy
import os
from scipy.spatial.transform import Rotation as R
from camera import Camera
from grasp_module import infer_grasps

matplotlib.use('Agg')

json_numpy.patch()

def setup_LMP(general_config, ur5, debug=False):
  planner_config = general_config['planner']
  lmp_env_config = general_config['lmp_config']['env']

  #修改lmps_config
  lmps_config = general_config['lmp_config']['lmps']
  env_name = general_config['env_name']
  # LMP env wrapper
  lmp_env = LMP_interface(lmp_env_config, planner_config, ur5, env_name=env_name)

  # creating the LMP that deals w/ high-level language commands
  task_planner = LMP(
      'planner', lmps_config, debug, env_name
  )

  return task_planner, lmp_env


class LMP_interface():

  def __init__(self, lmp_config, planner_config, ur5, env_name='rlbench'):
    self._env_name = env_name
    self._cfg = lmp_config
    self._map_size = self._cfg['map_size']
    self.slow_planner = Slow_PathPlanner(planner_config, map_size=self._map_size)
    self.fast_planner = Fast_PathPlanner(planner_config)
    self.camera = Camera()
    self.ur5 = ur5

    # calculate size of each voxel (resolution)
    self._resolution = (self.ur5.workspace_bounds_max - self.ur5.workspace_bounds_min) / self._map_size
    print(f'Voxel resolution: {self._resolution}')


  def get_obs(self, obj_pc, label, grasp_pose):
    obs_dict = dict()
    voxel_map = self._points_to_voxel_map(obj_pc)
    aabb_min = self._world_to_voxel(np.min(obj_pc, axis=0))
    aabb_max = self._world_to_voxel(np.max(obj_pc, axis=0))
    obs_dict['occupancy_map'] = voxel_map  # in voxel frame
    obs_dict['name'] = label
    obs_dict['position'] = self._world_to_voxel(np.mean(obj_pc, axis=0))  # in voxel frame
    obs_dict['aabb'] = np.array([aabb_min, aabb_max])  # in voxel frame
    obs_dict['_position_world'] = np.mean(obj_pc, axis=0)  # in world frame
    obs_dict['_point_cloud_world'] = obj_pc  # in world frame
    
    if grasp_pose :
      obs_dict['translation'] = self._world_to_voxel(grasp_pose['translation'])  # in world frame
      obs_dict['rotvec'] = grasp_pose['rotvec']  # in world frame

    object_obs = {"obs":Observation(obs_dict)}
    return object_obs

  def get_ee_obs(self):
      # 获取末端姿态
      obs_dict = dict()
      obs_dict['name'] = "gripper"
      obs_dict['position'] = self.get_ee_pos()
      obs_dict['aabb'] = np.array([self.get_ee_pos(), self.get_ee_pos()])
      obs_dict['_position_world'] = self.ur5.get_tcp()[:3]
      object_obs = {"obs":Observation(obs_dict)}
      return object_obs

  def get_table_obs(self):
      offset_percentage = 0.1
      x_min = self.ur5.workspace_bounds_min[0] + offset_percentage * (self.ur5.workspace_bounds_max[0] - self.ur5.workspace_bounds_min[0])
      x_max = self.ur5.workspace_bounds_max[0] - offset_percentage * (self.ur5.workspace_bounds_max[0] - self.ur5.workspace_bounds_min[0])
      y_min = self.ur5.workspace_bounds_min[1] + offset_percentage * (self.ur5.workspace_bounds_max[1] - self.ur5.workspace_bounds_min[1])
      y_max = self.ur5.workspace_bounds_max[1] - offset_percentage * (self.ur5.workspace_bounds_max[1] - self.ur5.workspace_bounds_min[1])
      table_max_world = np.array([x_max, y_max, 0])
      table_min_world = np.array([x_min, y_min, 0])
      table_center = (table_max_world + table_min_world) / 2
      obs_dict = dict()
      obs_dict['name'] = "workspace"
      obs_dict['position'] = self._world_to_voxel(table_center)
      obs_dict['_position_world'] = table_center
      obs_dict['aabb'] = np.array([self._world_to_voxel(table_min_world), self._world_to_voxel(table_max_world)])

      object_obs = {"obs":Observation(obs_dict)}
      return object_obs

  

  def transform_points_to_world(self, points_homogeneous, T_camera_to_world):
    """
    将形状为 (N, 3) 的点从相机坐标系变换到世界坐标系

    参数:
        points_homogeneous: shape (N, 3)，相机坐标系下的齐次点
        T_camera_to_world: shape (4, 4)，相机到世界的变换矩阵

    返回:
        points_world: shape (N, 3)，世界坐标系下的齐次点
    """
    # 获取点数量
    N = points_homogeneous.shape[0]

    # 添加齐次坐标 1，变成 (N, 4)
    points_homogeneous = np.hstack([
        points_homogeneous,
        np.ones((N, 1))
    ])  # shape (N, 4)

    # 应用变换
    points_world = T_camera_to_world @ points_homogeneous.T
    points_world = points_world.T[:, :3]

    return points_world

  def qwen_vl_box(self,instruction):
    rgb, img_depth = self.camera.get_aligned_images()
    image = Image.fromarray(np.array(rgb))
    image_path = f"tmp/images/rgb.jpeg"
    image.save(image_path)
    bbox = get_world_bboxs_list(image_path,instruction)
    return rgb, bbox
  
  def get_grasp_pose(self,color,meter_depth,workspace_mask,init,grasp_ids):
      if init:
        grasp_ids = [0]
        target_gg, grasp_ids = infer_grasps(color, meter_depth, workspace_mask, self.camera, init, grasp_ids)
        init = False
      else:
        target_gg, grasp_ids = infer_grasps(color, meter_depth, workspace_mask, self.camera, init, grasp_ids)

      grasp_pose = None
      gg_final = target_gg[0]
      T_gg_grasp = np.eye(4)
      T_gg_grasp[:3, :3] = gg_final.rotation_matrix
      T_gg_grasp[:3, 3] = gg_final.translation

      T_grasp2cam = np.eye(4)
      T_gg_cam = T_grasp2cam @ T_gg_grasp
      
      T_cam2world = self.camera.get_extrinsic_matrix()
      T_grasp2world = T_cam2world @ T_gg_cam
      
      rotation_matrix = R.from_matrix(T_grasp2world[:3, :3])
      rotvec = rotation_matrix.as_rotvec()
      translation = T_grasp2world[:3, 3]
      # 补偿gripper高度
      translation[2] = translation[2] + 0.2

      grasp_pose = {
        "translation": translation,
        "rotvec": rotvec
      }

      return grasp_pose, init, grasp_ids


  def update_mask_entities(self,instruction,finished_event,grasp_event,grasp_object,state_manager,init_grasp_finished):
      if not os.path.exists("tmp/images"):
          os.makedirs("tmp/images")
      if not os.path.exists("tmp/masks"):
          os.makedirs("tmp/masks")
      
      state = {}
      plt.figure(figsize=(20, 20))
      frame, bbox_entities = self.qwen_vl_box(instruction)
      print(bbox_entities)

      visuals,objects,label2id,id2label = process_visual_prompt(bbox_entities)
      set_visual_prompt(frame, visuals, objects)
      num = 0
      init = True
      grasp_ids = []
      while not finished_event.is_set():
        start_time = time.time()
        label_index = {}
        for item in objects:
          if item not in label_index.keys():
            label_index[item] = 1

        color, meter_depth = self.camera.get_aligned_images()
        # 这里创建的点云，原点为相机坐标系中心
        pcd_ = self.camera.create_point_cloud_from_depth_image(meter_depth)

        plt.clf()
        plt.imshow(frame)
        boxes, masks_ = predict_mask(frame)
        workspace_mask = np.zeros_like(meter_depth).astype(bool)
        for (box_ent, mask) in zip(boxes, masks_):
            id = int(box_ent[5])
            label = id2label[id]
            points, masks = [], []
            box = box_ent[:4]
            
            workspace_mask = workspace_mask | mask.astype(bool)
            
            points.append(pcd_.reshape(-1, 3))
            h, w = mask.shape[-2:]
            show_mask(mask,plt.gca())
            mask =  mask.reshape(h, w).reshape(-1)

            masks.append(mask)

            points, masks = np.array(points), np.array(masks)
            obj_points = points[np.isin(masks, 1)]

            # 这里将相机下的点云转换到世界坐标系下
            T_camera_to_base = self.camera.get_extrinsic_matrix()
            obj_points = self.transform_points_to_world(obj_points, T_camera_to_base)

            if len(obj_points) == 0:
                print(f"Scene not object {label}!")
                continue

            # voxel downsample using o3d
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(obj_points)
            pcd_downsampled = pcd.voxel_down_sample(voxel_size=0.001)
            obj_points = np.asarray(pcd_downsampled.points)

            grasp_pose = None
            # 如果有抓取事件，则进行抓取
            test1 = time.time()
            print("yoloe time: ",test1-start_time)
            if grasp_event.is_set():
              grasp_name = grasp_object.get()
              grasp_object.put(grasp_name)
              if grasp_name == label:
                print(f"Grasping {label}!")
                color = np.array(frame.copy(), dtype=np.float32) / 255.0
                meter_depth = np.array(meter_depth.copy(), dtype=np.float32)
                workspace_mask = workspace_mask.astype(bool)
                grasp_pose, init, grasp_ids = self.get_grasp_pose(color, meter_depth, workspace_mask, init, grasp_ids)
                init_grasp_finished.set()
                test2 = time.time()
                print("anygrasp time: ",test2-test1)
                
            obs = self.get_obs(obj_points, label, grasp_pose)
            state[label] = obs

            x_min, y_min, x_max, y_max = box
            center_x = (x_min + x_max) / 2
            center_y = (y_min + y_max) / 2

            # 在中心位置显示label
            plt.text(center_x, center_y, label, color='white', ha='center', va='center', fontsize=12, weight='bold')


        state['gripper'] = self.get_ee_obs()
        state['workspace'] = self.get_table_obs()

        state_manager.write_state(state)
        #print(state)

        end_time = time.time()  # 记录结束时间
        print(f"{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] updated object state in {end_time - start_time:.3f}s{bcolors.ENDC}")
        plt.axis('off')
        plt.draw()
        plt.savefig(f"tmp/masks/mask_{num}.jpeg", bbox_inches='tight', pad_inches=0)
        num+=1
  

  def get_ee_pos(self):
    return self._world_to_voxel(self.ur5.get_tcp()[:3])

  def reset_to_default_pose(self):
     self.ur5.reset_to_default_pose()
  
  # ======================================================
  # == helper functions
  # ======================================================
  def get_scene_3d_obs(self):
      color, meter_depth = self.camera.get_aligned_images()
      # 这里创建的点云，原点为相机坐标系中心
      pcd_ = self.camera.create_point_cloud_from_depth_image(meter_depth)

      points = np.array(pcd_.reshape(-1, 3))

      # 这里将相机下的点云转换到世界坐标系下
      T_camera_to_base = self.camera.get_extrinsic_matrix()
      points = self.transform_points_to_world(points, T_camera_to_base)

      # voxel downsample using o3d
      pcd = o3d.geometry.PointCloud()
      pcd.points = o3d.utility.Vector3dVector(points)
      pcd_downsampled = pcd.voxel_down_sample(voxel_size=0.001)
      points = np.asarray(pcd_downsampled.points)

      return points


  def _get_scene_collision_voxel_map(self):
    collision_points_world = self.get_scene_3d_obs()
    collision_voxel = self._points_to_voxel_map(collision_points_world)
    return collision_voxel
  
  def _world_to_voxel(self, world_xyz):
    _world_xyz = world_xyz.astype(np.float32)
    _voxels_bounds_robot_min = self.ur5.workspace_bounds_min.astype(np.float32)
    _voxels_bounds_robot_max = self.ur5.workspace_bounds_max.astype(np.float32)
    _map_size = self._map_size
    voxel_xyz = pc2voxel(_world_xyz, _voxels_bounds_robot_min, _voxels_bounds_robot_max, _map_size)
    return voxel_xyz

  def _voxel_to_world(self, voxel_xyz):
    _voxels_bounds_robot_min = self.ur5.workspace_bounds_min.astype(np.float32)
    _voxels_bounds_robot_max = self.ur5.workspace_bounds_max.astype(np.float32)
    _map_size = self._map_size
    world_xyz = voxel2pc(voxel_xyz, _voxels_bounds_robot_min, _voxels_bounds_robot_max, _map_size)
    return world_xyz

  def _points_to_voxel_map(self, points):
    """convert points in world frame to voxel frame, voxelize, and return the voxelized points"""
    _points = points.astype(np.float32)
    _voxels_bounds_robot_min = self.ur5.workspace_bounds_min.astype(np.float32)
    _voxels_bounds_robot_max = self.ur5.workspace_bounds_max.astype(np.float32)
    _map_size = self._map_size
    return pc2voxel_map(_points, _voxels_bounds_robot_min, _voxels_bounds_robot_max, _map_size)

  def _get_voxel_center(self, voxel_map):
    """calculte the center of the voxel map where value is 1"""
    voxel_center = np.array(np.where(voxel_map == 1)).mean(axis=1)
    return voxel_center

  def _get_default_voxel_map(self, type='target'):
    """returns default voxel map (defaults to current state)"""
    def fn_wrapper():
      if type == 'target':
        voxel_map = np.zeros((self._map_size, self._map_size, self._map_size))
      elif type == 'obstacle':  # for LLM to do customization
        voxel_map = np.zeros((self._map_size, self._map_size, self._map_size))
      elif type == 'velocity':
        voxel_map = np.ones((self._map_size, self._map_size, self._map_size))
      elif type == 'gripper':
        # 这里gripper:1->0为张开;0->1为闭合
        #voxel_map = np.ones((self._map_size, self._map_size, self._map_size)) * self._env.get_last_gripper_action()
        voxel_map = np.ones((self._map_size, self._map_size, self._map_size))
      elif type == 'rotation':
        voxel_map = np.zeros((self._map_size, self._map_size, self._map_size, 3))
        voxel_map[:, :, :] = self.ur5.get_tcp()[3:]
      else:
        raise ValueError('Unknown voxel map type: {}'.format(type))
      voxel_map = VoxelIndexingWrapper(voxel_map)
      return voxel_map
    return fn_wrapper

# ======================================================
# jit-ready functions (for faster replanning time, need to install numba and add "@njit")
# ======================================================
def pc2voxel(pc, voxel_bounds_robot_min, voxel_bounds_robot_max, map_size):
  """voxelize a point cloud"""
  pc = pc.astype(np.float32)
  # make sure the point is within the voxel bounds
  pc = np.clip(pc, voxel_bounds_robot_min, voxel_bounds_robot_max)
  # voxelize
  voxels = (pc - voxel_bounds_robot_min) / (voxel_bounds_robot_max - voxel_bounds_robot_min) * (map_size - 1)
  # to integer
  _out = np.empty_like(voxels)
  voxels = np.round(voxels, 0, _out).astype(np.int32)
  assert np.all(voxels >= 0), f'voxel min: {voxels.min()}'
  assert np.all(voxels < map_size), f'voxel max: {voxels.max()}'
  return voxels

def voxel2pc(voxels, voxel_bounds_robot_min, voxel_bounds_robot_max, map_size):
  """de-voxelize a voxel"""
  # check voxel coordinates are non-negative
  assert np.all(voxels >= 0), f'voxel min: {voxels.min()}'
  assert np.all(voxels < map_size), f'voxel max: {voxels.max()}'
  voxels = voxels.astype(np.float32)
  # de-voxelize
  pc = voxels / (map_size - 1) * (voxel_bounds_robot_max - voxel_bounds_robot_min) + voxel_bounds_robot_min
  return pc

def pc2voxel_map(points, voxel_bounds_robot_min, voxel_bounds_robot_max, map_size):
  """given point cloud, create a fixed size voxel map, and fill in the voxels"""
  points = points.astype(np.float32)
  voxel_bounds_robot_min = voxel_bounds_robot_min.astype(np.float32)
  voxel_bounds_robot_max = voxel_bounds_robot_max.astype(np.float32)
  # make sure the point is within the voxel bounds
  points = np.clip(points, voxel_bounds_robot_min, voxel_bounds_robot_max)
  # voxelize
  voxel_xyz = (points - voxel_bounds_robot_min) / (voxel_bounds_robot_max - voxel_bounds_robot_min) * (map_size - 1)
  # to integer
  _out = np.empty_like(voxel_xyz)
  points_vox = np.round(voxel_xyz, 0, _out).astype(np.int32)
  voxel_map = np.zeros((map_size, map_size, map_size))
  for i in range(points_vox.shape[0]):
      voxel_map[points_vox[i, 0], points_vox[i, 1], points_vox[i, 2]] = 1
  return voxel_map