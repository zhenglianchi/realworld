"""
脚本1: 仅使用VLM (Qwen2.5-VL-72B) 进行目标检测
输入文本指令 → VLM检测 → 显示目标框
每10s统计一次检测成功次数
"""
import cv2
import numpy as np
from PIL import Image
import time
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from camera import Camera
from VLM_demo import encode_image, smart_resize, resize_bbox_to_original, show_box_cv2
from openai import OpenAI
import json
from collections import deque


def vlm_detect(image_path, instruction):
    """调用VLM进行目标检测，输出bbox列表"""
    client = OpenAI(
        api_key="sk-df55df287b2c420285feb77137467576",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    base64_image = encode_image(image_path)

    prompt = f'''
    You are analyzing a robotic arm operation scene. Detect all objects in the image.

    For each detected element, output a dictionary with:
    - `"bbox"`: the 2D bounding box in format `[x1, y1, x2, y2]` (top-left and bottom-right),
    - `"label"`: the **English name** of the object.

    The user instruction is: "{instruction}"
    Focus on objects relevant to this instruction.

    Output **only a list of dictionaries**, no additional text or explanation.

    ### Output Format:
    ```json
    [
    {{"bbox": [x1, y1, x2, y2], "label": "white cube"}},
    {{"bbox": [x1, y1, x2, y2], "label": "glass cup"}}
    ]
    ```
    '''

    completion = client.chat.completions.create(
        model="qwen2.5-vl-72b-instruct",
        messages=[{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url",
             "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
             }
        ]}]
    )

    bbox_list_str = completion.choices[0].message.content[7:-3]
    bbox_list = json.loads(bbox_list_str)

    # 尺寸还原
    image = Image.open(image_path)
    w, h = image.size
    w_bar, h_bar = smart_resize(image_path)
    bbox_list_original = resize_bbox_to_original(bbox_list, (w, h), (w_bar, h_bar))

    return bbox_list_original


class DetectionStats:
    """10s滑动窗口统计检测成功次数"""
    def __init__(self, window=10.0):
        self.window = window
        self.records = deque()  # (timestamp, success: bool, latency: float)
        self.last_report_time = time.time()

    def add(self, success, latency):
        now = time.time()
        self.records.append((now, success, latency))
        # 清理超过窗口的旧记录
        while self.records and now - self.records[0][0] > self.window:
            self.records.popleft()

        # 每10s输出一次
        if now - self.last_report_time >= 10.0:
            successes = sum(1 for _, s, _ in self.records if s)
            latencies = [l for _, _, l in self.records]
            avg_lat = np.mean(latencies) if latencies else 0
            print(f"[LOG][VLM检测] 近10s统计: 检测{len(self.records)}次, "
                  f"成功{successes}次, 成功率{successes/len(self.records)*100:.1f}%, "
                  f"平均延时{avg_lat:.3f}s")
            self.last_report_time = now


def main():
    instruction = input("请输入检测指令 (例如: detect all objects): ").strip()
    if not instruction:
        instruction = "detect all objects"

    camera = Camera()
    print("正在初始化相机...")
    for _ in range(30):
        camera.get_aligned_images()

    stats = DetectionStats(window=10.0)
    print("按 'q' 退出, 按 's' 保存图片")

    while True:
        frame, _ = camera.get_aligned_images()
        image_path = "detection_scripts/tmp_vlm.jpg"
        os.makedirs("detection_scripts", exist_ok=True)
        Image.fromarray(frame).save(image_path)

        t0 = time.time()
        bbox_list = vlm_detect(image_path, instruction)
        vlm_time = time.time() - t0
        success = len(bbox_list) > 0
        stats.add(success, vlm_time)

        frame_draw = frame.copy()
        for entity in bbox_list:
            frame_draw = show_box_cv2(entity["bbox"], frame_draw, label=entity["label"],
                                       color=(0, 255, 0), thickness=2)

        cv2.imshow("VLM Detection", cv2.cvtColor(frame_draw, cv2.COLOR_RGB2BGR))
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite("detection_scripts/vlm_detect_result.jpg",
                        cv2.cvtColor(frame_draw, cv2.COLOR_RGB2BGR))
            print("已保存结果到 detection_scripts/vlm_detect_result.jpg")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
