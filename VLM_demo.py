from openai import OpenAI
import base64
from PIL import Image
import json
import math
import numpy as np
import requests
import json_numpy
import matplotlib.pyplot as plt
import time
import torch
import sys
import re
from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe.predict_vp import YOLOEVPSegPredictor
import io
import os
import cv2

_model = None

def get_model():
    global _model
    if _model is None:
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "yoloe-11s-seg.pt")
        _model = YOLOE(model_path).cuda()
    return _model

def process_visual_prompt(bbox_entities):
    bbox = np.array([item["bbox"] for item in bbox_entities])
    labels = [item["label"] for item in bbox_entities]
    cls_label = []
    cls = []
    classes = []
    label2id = {}
    index = 0
    for label in labels:
        # 去除末尾的数字并转换为统一类别
        key = re.sub(r'\d+$', '', label)
        cls_label.append(key)
        if key not in label2id:
            label2id[key] = index
            index += 1

    cls = np.array([label2id[label] for label in cls_label])
    classes = list(label2id.keys())

    visuals = dict(
        bboxes=[bbox]
        ,
        cls=[cls]
    )
    return visuals,classes,label2id,cls_label

def set_visual_prompt(source_image,prompts,classes):
    m = get_model()
    m.predict(source_image, prompts=prompts, predictor=YOLOEVPSegPredictor,return_vpe=True, save=False, verbose=False, imgsz=(480,640))
    m.set_classes(classes, m.predictor.vpe)
    m.predictor = None  # remove VPPredictor

def predict_mask(target_image):
    input_image = torch.from_numpy(target_image).permute(2, 0, 1).unsqueeze(0).float().cuda() / 255.0
    result = get_model().predict(input_image, save=False, conf=0.3, iou=0.05, verbose=False, imgsz=(480,640))
    if result[0].masks is None:
        #print("No mask detected!")
        return [], []
    masks = result[0].masks.data
    boxes = result[0].boxes.data
    return boxes.detach().cpu().numpy(), masks.detach().cpu().numpy()


def write_state(output_json_path,state,lock):
    while True:
        with lock:
            with open(output_json_path, 'w', encoding='utf-8') as json_file:
                json.dump(state, json_file)
                break
        time.sleep(0.1)
        

def read_state(state_json_path,lock):
    while True:
        with lock:
            with open(state_json_path, 'r', encoding='utf-8') as json_file:
                loaded_state = json.load(json_file)
                break
        time.sleep(0.1)

    return loaded_state

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
    
def encode_image_PIL(plt):
    # 创建一个内存中的字节流
    buf = io.BytesIO()
    # 将当前图像保存到内存字节流中，格式为 JPEG
    plt.savefig(buf, format='jpeg', bbox_inches='tight', pad_inches=0)
    # 获取字节数据
    image_bytes = buf.getvalue()
    # 关闭缓冲区
    buf.close()
    # 编码为 base64 字符串
    image_base64 = base64.b64encode(image_bytes).decode('utf-8')
    return image_base64

def resize_bbox_to_original(bbox_list, original_size, resized_size):
    # 获取原图和调整后图像的尺寸
    original_width, original_height = original_size
    resized_width, resized_height = resized_size
    
    # 计算缩放比例
    scale_x = original_width / resized_width
    scale_y = original_height / resized_height
    
    # 将边界框放大回原图像尺寸
    resized_bbox_list = []
    for detection in bbox_list:
        bbox = detection['bbox']
        
        # 放大边界框坐标
        x1, y1, x2, y2 = bbox
        new_x1 = int(x1 * scale_x)
        new_y1 = int(y1 * scale_y)
        new_x2 = int(x2 * scale_x)
        new_y2 = int(y2 * scale_y)
        
        # 更新到放大后的边界框列表
        resized_bbox_list.append({
            'bbox': [new_x1, new_y1, new_x2, new_y2],
            'label': detection['label']
        })
    
    return resized_bbox_list


def smart_resize(image_path, factor = 28, vl_high_resolution_images = False):
    # 打开指定的PNG图片文件
    image = Image.open(image_path)

    # 获取图片的原始尺寸
    height = image.height
    width = image.width
    # 将高度调整为28的整数倍
    h_bar = round(height / factor) * factor
    # 将宽度调整为28的整数倍
    w_bar = round(width / factor) * factor
    
    # 图像的Token下限：4个Token
    min_pixels = 28 * 28 * 4
    
    # 根据vl_high_resolution_images参数确定图像的Token上限
    if not vl_high_resolution_images:
        max_pixels = 1280 * 28 * 28
    else:
        max_pixels = 16384 * 28 * 28
        
    # 对图像进行缩放处理，调整像素的总数在范围[min_pixels,max_pixels]内
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return w_bar , h_bar


def get_world_bboxs_list(image_path,instruction):

    client = OpenAI(
        api_key="sk-df55df287b2c420285feb77137467576",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    base64_image = encode_image(image_path)

    prompt = f'''
    You are analyzing a robotic arm operation scene. Your task is to detect **all components** of objects associated with the instruction: "{instruction}".

    For each detected element, output a dictionary with:
    - `"bbox"`: the 2D bounding box in format `[x1, y1, x2, y2]` (top-left and bottom-right),
    - `"label"`: the **English name** of the part or object.

    ### Detection Rules:
    1. **Decompose composite objects into their visible parts**. For example:
    - *Cup* → "cup body", "cup handle"
    - *Cabinet* → "cabinet body", "cabinet door", "cabinet handle"
    - *Drawer* → "drawer front", "drawer body"
    - *Mug with handle* → "mug body", "handle"
    2. **Do not decompose inherently single-piece objects** — if an object has no meaningful detachable or distinguishable subparts, return it as a whole. Examples include:
    - "mouse", "pen", "block", "ball", "key", "usb drive"
    3. Only include **visually present and distinguishable** parts.
    4. Never return the full object if it has detectable components — **only return its parts**.
    5. If a part is occluded but still partially visible, include it with its visible bounding box.
    6. Output **only a list of dictionaries**, no additional text or explanation.
    7. Not to detect the robot part of the arm, only the objects and parts associated with the instruction.

    ### Output Format:
    ```json
    [
    {{"bbox": [x1, y1, x2, y2], "label": "cup handle"}},
    {{"bbox": [x1, y1, x2, y2], "label": "cup body"}},
    {{"bbox": [x1, y1, x2, y2], "label": "mouse"}},
    {{"bbox": [x1, y1, x2, y2], "label": "black block"}},
    {{"bbox": [x1, y1, x2, y2], "label": "white block"}},
    {{"bbox": [x1, y1, x2, y2], "label": "green block"}},
    {{"bbox": [x1, y1, x2, y2], "label": "cabinet door"}},
    {{"bbox": [x1, y1, x2, y2], "label": "teapot handle"}},
    {{"bbox": [x1, y1, x2, y2], "label": "teapot body"}},
    {{"bbox": [x1, y1, x2, y2], "label": "flower line foliage"}},
    {{"bbox": [x1, y1, x2, y2], "label": "flower bloom"}},
    {{"bbox": [x1, y1, x2, y2], "label": "drawer cup"}},
    {{"bbox": [x1, y1, x2, y2], "label": "drawer body"}},
    ]
    ```
    Ensure completeness and precision at the **part level**, respecting whether an object should be split or kept whole.

    '''

    completion = client.chat.completions.create(
        model="qwen2.5-vl-72b-instruct",
        messages=[{"role": "user","content": [
                {"type": "text","text": prompt},
                {"type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}, 
                }
                ]}]
        )

    bbox_list_str = completion.choices[0].message.content[7:-3]

    bbox_list = json.loads(bbox_list_str)

    # 打开图片
    image = Image.open(image_path)
    w , h = image.size

    w_bar,h_bar = smart_resize(image_path)

    bbox_list_orignal = resize_bbox_to_original(bbox_list, (w, h), (w_bar, h_bar))

    return bbox_list_orignal



def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2)) 

def show_mask(mask ,ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def get_response(url,query):
    response = requests.post(url, json=query)
    if response.status_code == 200:
        return response.json()
    else:
        print("Error:", response.status_code, response.text)

def show_mask_cv2(mask, image, random_color=False):
    if random_color:
        color = np.random.randint(0, 256, 3, dtype=np.uint8)
    else:
        color = np.array([30, 144, 255], dtype=np.uint8)  # dodgerblue
    h, w = mask.shape[-2:]
    mask = mask.reshape(h, w, 1)
    colored_mask = mask * color.reshape(1, 1, 3)
    
    # 叠加到原图
    image[mask[:, :, 0] > 0] = (
        image[mask[:, :, 0] > 0] * 0.5 + colored_mask[mask[:, :, 0] > 0] * 0.5
    ).astype(np.uint8)
    return image

def show_box_cv2(box, image, label=None, color=(0, 0, 255), thickness=1):
    x0, y0, x1, y1 = map(int, box)
    cv2.rectangle(image, (x0, y0), (x1, y1), color, thickness)
    if label:
        cv2.putText(image, label, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
    return image


def generate_augmented_prompts(original_image, bbox_entities, num_aug=3, min_crop=0.80, max_crop=0.95):
    """
    从VLM输出的检测框生成增强提示，包含原始框和num_aug个随机裁剪增强样本。

    对每个检测框：
    1. 裁剪出原始框区域
    2. 生成num_aug个随机裁剪，裁剪尺寸占原始框的[min_crop, max_crop]比例
    3. 随机偏移裁剪框位置，保持比例在范围内
    4. 将原始框和所有增强框合并为最终提示列表

    Args:
        original_image: PIL Image or numpy array, 原始完整图像
        bbox_entities: list[dict], 每个元素 {"bbox": [x1,y1,x2,y2], "label": "label"}
        num_aug: int, 每个原始框生成多少增强样本，默认3
        min_crop: float, 最小裁剪比例，默认0.80
        max_crop: float, 最大裁剪比例，默认0.95

    Returns:
        augmented_entities: list[dict], 增强后的实体列表，包含原始+所有增强
    """
    augmented_entities = []

    # 添加原始实体
    for entity in bbox_entities:
        # 原始框加入结果
        augmented_entities.append(entity.copy())
        x1, y1, x2, y2 = entity["bbox"]
        label = entity["label"]
        orig_w = x2 - x1
        orig_h = y2 - y1

        # 为每个原始框生成num_aug个随机裁剪增强
        for i in range(num_aug):
            # 随机采样裁剪比例
            crop_scale = np.random.uniform(min_crop, max_crop)
            crop_w = int(orig_w * crop_scale)
            crop_h = int(orig_h * crop_scale)

            # 随机采样偏移（保证裁剪框仍在原始框内）
            # 偏移范围: 0 ~ (orig - crop)
            dx = np.random.randint(0, orig_w - crop_w + 1)
            dy = np.random.randint(0, orig_h - crop_h + 1)

            # 计算新框坐标（相对于原图）
            new_x1 = x1 + dx
            new_y1 = y1 + dy
            new_x2 = new_x1 + crop_w
            new_y2 = new_y1 + crop_h

            # 添加增强样本，标签相同
            augmented_entities.append({
                "bbox": [new_x1, new_y1, new_x2, new_y2],
                "label": label
            })

    return augmented_entities


def compute_motion_vectors(prev_entities, curr_entities):
    """
    计算每个物体中心在相邻两次成功检测之间的运动向量（像素位移）。

    Args:
        prev_entities / curr_entities: list[dict]，元素为 {"bbox":[x1,y1,x2,y2], "label": str}
            分别对应上一帧与当前帧的检测结果。
    Returns:
        motion: dict，key 为去除尾部数字后的类别名，value 为该类别物体中心的位移向量
                (curr_center - prev_center)；当前帧新出现、上一帧没有的类别返回零向量。
    """
    motion = {}

    def bbox_center(box):
        return np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])

    prev_dict = {}
    for ent in prev_entities:
        key = re.sub(r'\d+$', '', ent["label"])
        if key not in prev_dict:
            prev_dict[key] = []
        prev_dict[key].append(bbox_center(ent["bbox"]))

    curr_dict = {}
    for ent in curr_entities:
        key = re.sub(r'\d+$', '', ent["label"])
        if key not in curr_dict:
            curr_dict[key] = []
        curr_dict[key].append(bbox_center(ent["bbox"]))

    for label in curr_dict:
        if label in prev_dict:
            prev_center = np.mean(prev_dict[label], axis=0)
            curr_center = np.mean(curr_dict[label], axis=0)
            motion[label] = curr_center - prev_center
        else:
            motion[label] = np.array([0.0, 0.0])
    return motion


def generate_motion_augmented_prompts(bbox_entities, motion_vectors, num_aug=3,
                                      min_crop=0.80, max_crop=0.95, sensitivity=1.0):
    """
    运动偏置空间裁剪（Motion-biased Spatial Cropping），实现与论文公式一致。

    对每个原始检测框（宽 w、高 h），生成 num_aug 个增广提示：
        1. 裁剪比例均匀采样：s_k ~ U(min_crop, max_crop)，缩放后宽高 w' = s_k*w, h' = s_k*h；
        2. 沿运动方向的位移（sensitivity 即论文中的灵敏度系数 λ）：
               δ_x = λ · clip(d_x, −w'/2,  w'/2)
               δ_y = λ · clip(d_y, −h'/2,  h'/2)
           其中 (d_x, d_y) 为相邻两次检测的物体包围盒中心位移（motion_vectors 提供）；
        3. 增广框左上角偏移从运动偏置分布中采样：
               Δx_k ~ U( max(0, δ_x), min(w − w', w − w' + δ_x) )
               Δy_k ~ U( max(0, δ_y), min(h − h', h − h' + δ_y) )
           即裁剪窗口偏向物体运动方向一侧，使增广视图包含运动一致的视觉内容。
    当物体无明显运动（δ≈0）时，上述分布退化为框内均匀随机裁剪（与 +Random 变体一致）。

    Args:
        bbox_entities: list[dict]，VLM 输出的原始检测框，元素 {"bbox":[x1,y1,x2,y2], "label": str}
        motion_vectors: dict，类别名 -> 2 维位移向量（像素），由 compute_motion_vectors 得到
        num_aug: int, 每个原始框生成的增广样本数 K（默认 3）
        min_crop / max_crop: float, 裁剪比例 s_k 的采样区间
        sensitivity: float, 灵敏度系数 λ（论文 eq. δ_x = λ·clip(d_x, −w'/2, w'/2)），默认 1.0

    Returns:
        augmented_entities: list[dict]，原始框 + 全部增广框
    """
    augmented_entities = []

    for entity in bbox_entities:
        # 原始框保留
        augmented_entities.append(entity.copy())
        x1, y1, x2, y2 = entity["bbox"]
        label = entity["label"]
        w = x2 - x1
        h = y2 - y1

        clean_label = re.sub(r'\d+$', '', label)
        motion_vec = motion_vectors.get(clean_label, np.array([0.0, 0.0]))
        d_x = float(motion_vec[0])
        d_y = float(motion_vec[1])

        for _ in range(num_aug):
            # 1) 裁剪比例 s_k ~ U(min_crop, max_crop)，缩放尺寸 w' = s_k*w, h' = s_k*h
            crop_scale = np.random.uniform(min_crop, max_crop)
            crop_w = int(w * crop_scale)
            crop_h = int(h * crop_scale)
            if crop_w <= 0 or crop_h <= 0 or crop_w >= w or crop_h >= h:
                continue

            # 2) 运动方向位移：δ_x = λ·clip(d_x, −w'/2, w'/2)，δ_y 同理
            delta_x = sensitivity * float(np.clip(d_x, -crop_w / 2.0, crop_w / 2.0))
            delta_y = sensitivity * float(np.clip(d_y, -crop_h / 2.0, crop_h / 2.0))

            # 裁剪框必须保持在原始框内：可行偏移范围为 [0, w−w']（[0, h−h']）
            margin_x = w - crop_w
            margin_y = h - crop_h
            # 当 |δ| 超出可行余量时夹取，保证下面的采样区间非空
            delta_x = float(np.clip(delta_x, -margin_x, margin_x))
            delta_y = float(np.clip(delta_y, -margin_y, margin_y))

            # 3) 从运动偏置分布采样左上角偏移
            #    lo = max(0, δ), hi = min(w−w', w−w'+δ)
            lo_x = max(0.0, delta_x)
            hi_x = margin_x + min(0.0, delta_x)
            lo_y = max(0.0, delta_y)
            hi_y = margin_y + min(0.0, delta_y)
            dx = int(round(np.random.uniform(lo_x, hi_x)))
            dy = int(round(np.random.uniform(lo_y, hi_y)))
            # 四舍五入越界保护
            dx = min(max(dx, 0), margin_x)
            dy = min(max(dy, 0), margin_y)

            new_x1 = x1 + dx
            new_y1 = y1 + dy
            new_x2 = new_x1 + crop_w
            new_y2 = new_y1 + crop_h

            augmented_entities.append({
                "bbox": [new_x1, new_y1, new_x2, new_y2],
                "label": label
            })

    return augmented_entities