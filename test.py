from camera import Camera
from grasp_module import infer_grasps
import numpy as np
import matplotlib.pyplot as plt


camera = Camera()
color, depth = camera.get_aligned_images()
init = True
grasp_ids = [0]
plt.figure(figsize=(20, 20))
plt.imshow(color)
plt.show()

color = np.array(color.copy(), dtype=np.float32) / 255.0
depth = np.array(depth.copy(), dtype=np.float32) / 1000.0

# workspace_mask需要判断curr_gg的trasnlation范围，而不是二维mask
workspace_mask = np.ones_like(depth).astype(bool)
target_gg, grasp_ids = infer_grasps(color, depth, workspace_mask, camera, init, grasp_ids)
print(grasp_ids)
print(target_gg)

