"""
脚本2: 使用 OWL-ViT 进行目标检测
输入文本查询 → OWL-ViT检测 → 显示目标框
每10s统计一次检测成功次数
"""
import cv2
import numpy as np
from PIL import Image
import time
import sys
import os
import torch
from collections import deque
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from camera import Camera
from transformers import OwlViTProcessor, OwlViTForObjectDetection


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
            print(f"[LOG][OWL-ViT检测] 近10s统计: 检测{len(self.records)}次, "
                  f"成功{successes}次, 成功率{successes/len(self.records)*100:.1f}%, "
                  f"平均延时{avg_lat:.3f}s")
            self.last_report_time = now


def main():
    print("正在加载 OWL-ViT 模型...")
    processor = OwlViTProcessor.from_pretrained(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "owlvit-base-patch16"))
    model = OwlViTForObjectDetection.from_pretrained(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "owlvit-base-patch16"))

    texts_input = input("请输入检测文本 (多个目标用逗号分隔, 例如: a photo of a cup, a photo of a block): ").strip()
    if not texts_input:
        texts_input = "a photo of a cup, a photo of a block"
    texts = [[t.strip() for t in texts_input.split(",")]]

    camera = Camera()
    print("正在初始化相机...")
    for _ in range(30):
        camera.get_aligned_images()

    stats = DetectionStats(window=10.0)
    print(f"查询文本: {texts[0]}")
    print("按 'q' 退出, 按 's' 保存图片")

    while True:
        frame, _ = camera.get_aligned_images()
        image = Image.fromarray(frame)

        t0 = time.time()
        inputs = processor(text=texts, images=image, return_tensors="pt")
        outputs = model(**inputs)
        target_sizes = torch.Tensor([image.size[::-1]])
        results = processor.post_process_object_detection(
            outputs=outputs, threshold=0.1, target_sizes=target_sizes)
        owl_time = time.time() - t0

        boxes, scores, labels = results[0]["boxes"], results[0]["scores"], results[0]["labels"]
        success = len(boxes) > 0
        stats.add(success, owl_time)

        frame_draw = frame.copy()
        for box, score, label in zip(boxes, scores, labels):
            box = [int(round(i)) for i in box.tolist()]
            label_name = texts[0][label]
            display_text = f"{label_name}: {score:.2f}"
            x0, y0, x1, y1 = box
            cv2.rectangle(frame_draw, (x0, y0), (x1, y1), (0, 255, 0), 2)
            cv2.putText(frame_draw, display_text, (x0, y0 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.imshow("OWL-ViT Detection", cv2.cvtColor(frame_draw, cv2.COLOR_RGB2BGR))
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite("detection_scripts/owlvit_detect_result.jpg",
                        cv2.cvtColor(frame_draw, cv2.COLOR_RGB2BGR))
            print("已保存结果到 detection_scripts/owlvit_detect_result.jpg")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
