import pyrealsense2 as rs
import numpy as np
from PIL import Image
import cv2

class Camera:
    def __init__(self):
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)

        self.pipeline.start(config)

        color_sensor = self.pipeline.get_active_profile().get_device().query_sensors()[1]  # 0-depth(两个infra)相机, 1-rgb相机,2-IMU
        # 自动曝光设置
        color_sensor.set_option(rs.option.enable_auto_exposure, True)

        align_to = rs.stream.color
        self.align = rs.align(align_to)

        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        aligned_depth_frame = aligned_frames.get_depth_frame()
        aligned_color_frame = aligned_frames.get_color_frame()
        depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics
        color_intrin = aligned_color_frame.profile.as_video_stream_profile().intrinsics

        self.depth_intrin = depth_intrin
        self.color_intrin = color_intrin
        
        self.width = self.color_intrin.width
        self.height = self.color_intrin.height
        self.fx = self.color_intrin.fx
        self.fy = self.color_intrin.fy
        self.cx = self.color_intrin.ppx
        self.cy = self.color_intrin.ppy
        self.scale = 1.0

        for i in range(10):
            self.get_aligned_images()


    def create_point_cloud_from_depth_image(self, depth):
        assert(depth.shape[0] == self.height and depth.shape[1] == self.width)
        xmap = np.arange(self.width)
        ymap = np.arange(self.height)
        xmap, ymap = np.meshgrid(xmap, ymap)
        points_z = depth / self.scale
        points_x = (xmap - self.cx) * points_z / self.fx
        points_y = (ymap - self.cy) * points_z / self.fy
        points = np.stack([points_x, points_y, points_z], axis=-1)
        return points

    def get_aligned_images(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        aligned_depth_frame = aligned_frames.get_depth_frame()
        aligned_color_frame = aligned_frames.get_color_frame()
        img_color = np.asanyarray(aligned_color_frame.get_data())
        img_depth = np.asanyarray(aligned_depth_frame.get_data())

        return img_color, img_depth


