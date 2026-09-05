"""
脚本4: VLM + YOLOE + 图像增强 (项目原方法)
VLM生成框 → 随机裁剪增强 → YOLOE视觉提示 → 实时检测
每10s统计一次检测成功次数
"""
import cv2
import numpy as np
from PIL import Image
import time
import sys
import os
import json
import torch
from collections import deque
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from camera import Camera
from openai import OpenAI
from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe.predict_vp import YOLOEVPSegPredictor
from VLM_demo import (encode_image, smart_resize, resize_bbox_to_original,
                       show_box_cv2, show_mask_cv2, generate_augmented_prompts)


def vlm_get_bboxes(image_path, instruction):
    """VLM生成检测框"""
    client = OpenAI(
        api_key="sk-df55df287b2c420285feb77137467576",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    base64_image = encode_image(image_path)

    prompt = f'''
    You are analyzing a robotic arm operation scene. Your task is to detect **all components** of objects associated with: "{instruction}".

    For each detected element, output:
    - `"bbox"`: `[x1, y1, x2, y2]`
    - `"label"`: the **English name** of the part or object.

    Decompose composite objects into visible parts.
    Output **only a list of dictionaries**, no additional text.

    ```json
    [
    {{"bbox": [x1, y1, x2, y2], "label": "cup body"}}
    ]
    ```
    '''

    result = client.chat.completions.create(
        model="qwen2.5-vl-72b-instruct",
        messages=[{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url",
             "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
             }
        ]}]
    )

    bbox_list_str = result.choices[0].message.content[7:-3]
    bbox_list = json.loads(bbox_list_str)

    image = Image.open(image_path)
    w, h = image.size
    w_bar, h_bar = smart_resize(image_path)
    bbox_list_original = resize_bbox_to_original(bbox_list, (w, h), (w_bar, h_bar))

    return bbox_list_original


def process_visual_prompt(bbox_entities):
    """处理视觉提示"""
    import re
    bbox = np.array([item["bbox"] for item in bbox_entities])
    labels = [item["label"] for item in bbox_entities]
    cls_label = []
    label2id = {}
    index = 0
    for label in labels:
        key = re.sub(r'\d+$', '', label)
        cls_label.append(key)
        if key not in label2id:
            label2id[key] = index
            index += 1

    cls = np.array([label2id[label] for label in cls_label])
    classes = list(label2id.keys())

    visuals = dict(
        bboxes=[bbox],
        cls=[cls]
    )
    return visuals, classes, label2id, cls_label


class DetectionStats:
    """10s滑动窗口统计检测成功次数"""
    def __init__(self, window=10.0):
        self.window = window
        self.records = deque()
        self.last_report_time = time.time()

    def add(self, success, latency):
        now = time.time()
        self.records.append((now, success, latency))
        while self.records and now - self.records[0][0] > self.window:
            self.records.popleft()

        if now - self.last_report_time >= 10.0:
            successes = sum(1 for _, s, _ in self.records if s)
            latencies = [l for _, _, l in self.records]
            avg_lat = np.mean(latencies) if latencies else 0
            print(f"[LOG][YOLOE检测] 近10s统计: 检测{len(self.records)}次, "
                  f"成功{successes}次, 成功率{successes/len(self.records)*100:.1f}%, "
                  f"平均延时{avg_lat:.3f}s")
            self.last_report_time = now


def main():
    instruction = input("请输入检测指令: ").strip()
    if not instruction:
        instruction = "detect all objects"

    # 加载YOLOE模型
    print("正在加载 YOLOE 模型...")
    model = YOLOE(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "yoloe-11s-seg.pt")).cuda()

    camera = Camera()
    print("正在初始化相机...")
    for _ in range(30):
        camera.get_aligned_images()

    # 第一帧 VLM 生成框
    frame, _ = camera.get_aligned_images()
    image_path = "detection_scripts/tmp_vlm_yoloe_aug.jpg"
    os.makedirs("detection_scripts", exist_ok=True)
    Image.fromarray(frame).save(image_path)

    t0 = time.time()
    bbox_entities = vlm_get_bboxes(image_path, instruction)
    vlm_time = time.time() - t0
    print(f"[LOG][VLM推理] VLM检测框生成延时: {vlm_time:.3f}s, 原始框数: {len(bbox_entities)}")

    # 数据增强
    t0 = time.time()
    augmented_bbox_entities = generate_augmented_prompts(
        frame, bbox_entities, num_aug=3, min_crop=0.80, max_crop=0.95
    )
    aug_time = time.time() - t0
    print(f"[LOG][数据增强] 增强后总框数: {len(augmented_bbox_entities)} "
          f"(原始{len(bbox_entities)} + 增强{len(augmented_bbox_entities) - len(bbox_entities)}), "
          f"耗时: {aug_time*1000:.1f}ms")

    # 设置YOLOE视觉提示
    t0 = time.time()
    visuals, classes, label2id, cls_label = process_visual_prompt(augmented_bbox_entities)
    model.predict(frame, prompts=visuals, predictor=YOLOEVPSegPredictor,
                  return_vpe=True, save=False, verbose=False, imgsz=(480, 640))
    model.set_classes(classes, model.predictor.vpe)
    model.predictor = None
    id2label = {v: k for k, v in label2id.items()}
    vp_time = time.time() - t0
    print(f"[LOG][YOLOE] 视觉提示设置延时: {vp_time:.3f}s")

    stats = DetectionStats(window=10.0)
    print("[LOG][开始计数] VLM提示已送入YOLOE模型，开始10s统计窗口")
    print("开始实时检测, 按 'q' 退出, 按 's' 保存图片")

    while True:
        frame, _ = camera.get_aligned_images()
        frame_draw = frame.copy()

        t0 = time.time()
        input_image = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float().cuda() / 255.0
        result = model.predict(input_image, save=False, conf=0.3, iou=0.05,
                               verbose=False, imgsz=(480, 640))
        yoloe_time = time.time() - t0

        if result[0].masks is not None:
            masks = result[0].masks.data.detach().cpu().numpy()
            boxes = result[0].boxes.data.detach().cpu().numpy()
            success = len(boxes) > 0
            stats.add(success, yoloe_time)

            for box_entity, mask in zip(boxes, masks):
                box = box_entity[:4]
                label_id = int(box_entity[5])
                label = id2label[label_id]
                frame_draw = show_mask_cv2(mask, frame_draw)
                frame_draw = show_box_cv2(box, frame_draw, label=label, thickness=2)
        else:
            stats.add(False, yoloe_time)

        cv2.imshow("VLM+YOLOE+Aug", cv2.cvtColor(frame_draw, cv2.COLOR_RGB2BGR))
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite("detection_scripts/vlm_yoloe_aug_result.jpg",
                        cv2.cvtColor(frame_draw, cv2.COLOR_RGB2BGR))
            print("已保存结果到 detection_scripts/vlm_yoloe_aug_result.jpg")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
