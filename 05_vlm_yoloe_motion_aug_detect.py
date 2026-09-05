"""
脚本5: VLM + YOLOE + 运动感知增强 (改进方法)
第1、2次检测 → 随机裁剪增强 (同脚本4)
第3次检测开始 → 用最近两次成功检测的包围盒中心位移估计运动方向，
                  按论文公式做运动偏置空间裁剪：
                      δ_x = λ·clip(d_x, −w'/2, w'/2)
                      Δx_k ~ U(max(0, δ_x), min(w−w', w−w'+δ_x))
                  裁剪窗口偏向运动方向一侧，保留运动一致的视觉内容。
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
import re
from collections import deque
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from camera import Camera
from openai import OpenAI
from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe.predict_vp import YOLOEVPSegPredictor
from VLM_demo import (encode_image, smart_resize, resize_bbox_to_original,
                      show_box_cv2, show_mask_cv2, generate_augmented_prompts,
                      compute_motion_vectors, generate_motion_augmented_prompts)


def vlm_get_bboxes(image_path, instruction):
    """VLM生成检测框"""
    client = OpenAI(
        api_key="sk-df55df287b2c420285feb77137467576",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    base64_image = encode_image(image_path)

    prompt = f'''
    You are analyzing a robotic arm operation scene. Detect **all components** of objects
    associated with: "{instruction}".

    For each detected element, output: `"bbox"`: `[x1, y1, x2, y2]`, `"label"`: English name.
    Decompose composite objects into visible parts.

    ```json
    [
    {{"bbox": [x1, y1, x2, y2], "label": "cup body"}}
    ]
    ```
    Output **only a list of dictionaries**.
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
    return resize_bbox_to_original(bbox_list, (w, h), (w_bar, h_bar))


# =====================================================================
# 运动估计与运动偏置空间裁剪已统一实现在 VLM_demo.py 中（与论文公式一致）：
#   - compute_motion_vectors(prev_entities, curr_entities):
#       相邻两次成功检测的同类包围盒中心位移 (d_x, d_y)
#   - generate_motion_augmented_prompts(bbox_entities, motion_vectors, ...):
#       δ_x = λ·clip(d_x, −w'/2, w'/2); Δx_k ~ U(max(0,δ_x), min(w−w', w−w'+δ_x))
# 本脚本直接复用，不再维护本地副本。
# =====================================================================


def process_visual_prompt(bbox_entities):
    """处理视觉提示"""
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

    visuals = dict(bboxes=[bbox], cls=[cls])
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


def reinit_yoloe_prompts(model, frame, bbox_entities, motion_vectors=None, use_motion=False):
    """重新设置 YOLOE 视觉提示。
    motion_vectors=None → 随机增强 (等同脚本4)
    motion_vectors 非空 → 运动感知增强（论文公式版，裁剪比例统一为 0.80–0.95，λ=1.0）
    """
    if use_motion and motion_vectors is not None:
        augmented_bbox = generate_motion_augmented_prompts(
            bbox_entities, motion_vectors, num_aug=3,
            min_crop=0.80, max_crop=0.95, sensitivity=1.0)
    else:
        augmented_bbox = generate_augmented_prompts(
            frame, bbox_entities, num_aug=3, min_crop=0.80, max_crop=0.95)

    visuals, classes, label2id, cls_label = process_visual_prompt(augmented_bbox)
    model.predictor = None  # 清掉上次检测残留的SegmentationPredictor，确保重新创建YOLOEVPSegPredictor
    model.predict(frame, prompts=visuals, predictor=YOLOEVPSegPredictor,
                  return_vpe=True, save=False, verbose=False, imgsz=(480, 640))
    model.set_classes(classes, model.predictor.vpe)
    model.predictor = None
    id2label = {v: k for k, v in label2id.items()}
    return id2label, augmented_bbox


def main():
    instruction = input("请输入检测指令: ").strip()
    if not instruction:
        instruction = "detect all objects"

    # 加载YOLOE模型
    print("正在加载 YOLOE 模型...")
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "yoloe-11s-seg.pt")
    model = YOLOE(model_path).cuda()

    camera = Camera()
    print("正在初始化相机...")
    for _ in range(30):
        camera.get_aligned_images()

    os.makedirs("detection_scripts", exist_ok=True)

    # ========= 阶段1: VLM初始检测 =========
    frame, _ = camera.get_aligned_images()
    image_path = "detection_scripts/tmp_motion_aug.jpg"
    Image.fromarray(frame).save(image_path)

    t0 = time.time()
    bbox_entities = vlm_get_bboxes(image_path, instruction)
    vlm_time = time.time() - t0
    print(f"[LOG][VLM推理] VLM检测框生成延时: {vlm_time:.3f}s, 框数: {len(bbox_entities)}")

    # ========= 阶段2: 初始设置: 随机增强 (第1、2次检测用) =========
    t0 = time.time()
    id2label, augmented_bbox = reinit_yoloe_prompts(
        model, frame, bbox_entities, motion_vectors=None, use_motion=False)
    init_time = time.time() - t0
    print(f"[LOG][初始增强] 随机增强(同脚本4): 总框数: {len(augmented_bbox)}, 延时: {init_time:.3f}s")
    print("[LOG][策略] 第1、2次检测使用随机裁剪增强, 第3次起使用运动感知增强")

    # ========= 阶段3: 实时检测循环 =========
    stats = DetectionStats(window=10.0)
    print("[LOG][开始计数] VLM提示已送入YOLOE模型，开始10s统计窗口")
    print("开始实时检测, 按 'q' 退出, 按 's' 保存, 按 'u' 重新VLM检测")
    detection_history = []  # 保存最近两次成功检测结果 [(boxes_list, time), ...]
    detection_count = 0     # 有结果的成功检测次数
    use_motion_mode = False   # 当前是否处于运动感知模式

    while True:
        frame, _ = camera.get_aligned_images()
        frame_draw = frame.copy()

        # YOLOE检测
        t0 = time.time()
        input_image = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float().cuda() / 255.0
        result = model.predict(input_image, save=False, conf=0.3, iou=0.05,
                               verbose=False, imgsz=(480, 640))
        yoloe_time = time.time() - t0

        curr_detected_boxes = []
        if result[0].masks is not None:
            masks = result[0].masks.data.detach().cpu().numpy()
            boxes = result[0].boxes.data.detach().cpu().numpy()
            stats.add(True, yoloe_time)
            detection_count += 1

            for box_entity, mask in zip(boxes, masks):
                box = box_entity[:4]
                label_id = int(box_entity[5])
                label = id2label[label_id]
                curr_detected_boxes.append({"bbox": [float(box[0]), float(box[1]),
                                                     float(box[2]), float(box[3])],
                                             "label": label})
                frame_draw = show_mask_cv2(mask, frame_draw)
                frame_draw = show_box_cv2(box, frame_draw, label=label, thickness=2)
        else:
            stats.add(False, yoloe_time)

        # ===== 运动感知增强逻辑 =====
        # 前几次成功检测: 随机增强 (已在阶段2设置) → 仅累积历史
        # 到第3次成功检测时: 用最近两次成功检测（history 倒数第2、1条）
        #   的包围盒中心位移估计运动方向, 切换为运动感知增强
        if len(curr_detected_boxes) > 0:
            detection_history.append((curr_detected_boxes, time.time()))
            # 只保留最近3次 (用于运动计算)
            if len(detection_history) > 3:
                detection_history.pop(0)

        # 第3次成功检测: 首次切换到运动感知模式
        if detection_count == 3 and not use_motion_mode and len(detection_history) >= 2:
            t_motion = time.time()
            motion_vectors = compute_motion_vectors(
                detection_history[-2][0],   # 最近第2次成功检测
                detection_history[-1][0]    # 最近第1次（当前）成功检测
            )
            motion_time = time.time() - t_motion

            t_reaug = time.time()
            id2label, augmented_bbox = reinit_yoloe_prompts(
                model, frame, bbox_entities, motion_vectors=motion_vectors, use_motion=True)
            reaug_time = time.time() - t_reaug

            use_motion_mode = True
            active_motions = {k: v for k, v in motion_vectors.items()
                              if np.linalg.norm(v) > 1}
            if active_motions:
                motion_info = " | ".join([f"{k}: ({v[0]:+.1f},{v[1]:+.1f})px"
                                           for k, v in active_motions.items()])
            else:
                motion_info = "无显著运动"
            print(f"[LOG][运动感知] 第3次检测, 切换到运动感知增强! "
                  f"运动计算: {motion_time*1000:.1f}ms, 重增强: {reaug_time*1000:.1f}ms, "
                  f"运动: {motion_info}")

        # 后续每10次成功检测更新一次运动方向
        if use_motion_mode and detection_count > 3 and detection_count % 10 == 0 and len(detection_history) >= 2:
            t_motion = time.time()
            motion_vectors = compute_motion_vectors(
                detection_history[-2][0],
                detection_history[-1][0]
            )
            motion_time = time.time() - t_motion

            t_reaug = time.time()
            id2label, augmented_bbox = reinit_yoloe_prompts(
                model, frame, bbox_entities, motion_vectors=motion_vectors, use_motion=True)
            reaug_time = time.time() - t_reaug

            active_motions = {k: v for k, v in motion_vectors.items()
                              if np.linalg.norm(v) > 1}
            if active_motions:
                motion_info = " | ".join([f"{k}: ({v[0]:+.1f},{v[1]:+.1f})px"
                                           for k, v in active_motions.items()])
                print(f"[LOG][运动感知] 更新运动方向: {motion_info}, "
                      f"计算: {motion_time*1000:.1f}ms, 重增强: {reaug_time*1000:.1f}ms")

        # 显示当前模式
        mode_text = "MOTION-AWARE" if use_motion_mode else "RANDOM (as script4)"
        cv2.putText(frame_draw, f"Mode: {mode_text} | Det#: {detection_count}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("VLM+YOLOE+Motion Aug (Improved)", cv2.cvtColor(frame_draw, cv2.COLOR_RGB2BGR))
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite("detection_scripts/vlm_yoloe_motion_aug_result.jpg",
                        cv2.cvtColor(frame_draw, cv2.COLOR_RGB2BGR))
            print("已保存结果")
        elif key == ord('u'):
            # 重新VLM检测, 重置为随机增强, 重新积累检测历史
            Image.fromarray(frame).save(image_path)
            t_vlm = time.time()
            bbox_entities = vlm_get_bboxes(image_path, instruction)
            vlm_time = time.time() - t_vlm
            print(f"[LOG][VLM推理] 重新检测延时: {vlm_time:.3f}s, 框数: {len(bbox_entities)}")
            # 重置: 回到随机增强模式
            id2label, augmented_bbox = reinit_yoloe_prompts(
                model, frame, bbox_entities, motion_vectors=None, use_motion=False)
            detection_history.clear()
            detection_count = 0
            use_motion_mode = False
            print("[LOG][策略] 已重置, 前2次检测将使用随机增强")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
