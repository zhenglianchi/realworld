import os
import argparse
import numpy as np
import open3d as o3d
from PIL import Image
from graspnetAPI import GraspGroup
import sys

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'anygrasp_sdk/grasp_tracking'))
sys.path.append(os.path.join(ROOT_DIR, 'anygrasp_sdk/pointnet2'))

from tracker import AnyGraspTracker

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', default="checkpoint_tracking.tar", help='Model checkpoint path')
parser.add_argument('--filter', type=str, default='oneeuro', help='Filter to smooth grasp parameters(rotation, width, depth). [oneeuro/kalman/none]')
cfgs = parser.parse_args()

# intialization
anygrasp_tracker = AnyGraspTracker(cfgs)
anygrasp_tracker.load_net()

def get_data(color,depth,camera):
    # get point cloud
    points = camera.create_point_cloud_from_depth_image(depth)
    mask = (points[:,:,2] >= 0) & (points[:,:,2] < 1)
    points = points[mask]
    colors = color[mask]

    return points, colors

def vis_grasps(gg, points, colors):
    grippers = gg.to_open3d_geometry_list()
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points.astype(np.float32))
    cloud.colors = o3d.utility.Vector3dVector(colors.astype(np.float32))
    o3d.visualization.draw_geometries([cloud, *grippers])


def infer_grasps(color,depth,workspace_mask,camera, init, grasp_ids):
    points, colors = get_data(color,depth,camera)

    target_gg, curr_gg, target_grasp_ids, corres_preds = anygrasp_tracker.update(points, colors, grasp_ids)
    if init:
        # 这里的mask是手动设置的，可以根据实际情况调整
        grasp_mask_x = ((curr_gg.translations[:,0]>-0.18) & (curr_gg.translations[:,0]<0.18))
        grasp_mask_y = ((curr_gg.translations[:,1]>-0.12) & (curr_gg.translations[:,1]<0.12))
        grasp_mask_z = ((curr_gg.translations[:,2]>0.35) & (curr_gg.translations[:,2]<0.55))
        workspace_mask = grasp_mask_x & grasp_mask_y & grasp_mask_z
        grasp_ids = np.where(workspace_mask)[0][:24:6]

        target_gg = curr_gg[grasp_ids]
    else:
        grasp_ids = target_grasp_ids

    vis_grasps(target_gg, points, colors)

    return target_gg, grasp_ids