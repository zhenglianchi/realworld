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

model = YOLOE("yoloe-11s-seg.pt").cuda()

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
    model.predict(source_image, prompts=prompts, predictor=YOLOEVPSegPredictor,return_vpe=True, save=False, verbose=False, imgsz=(480,640))
    model.set_classes(classes, model.predictor.vpe)
    model.predictor = None  # remove VPPredictor

def predict_mask(target_image):
    result = model.predict(target_image, save=False, conf=0.5, iou=0.5, verbose=False, imgsz=(480,640))
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


def get_world_bboxs_list(image_path,objects):

    client = OpenAI(
        api_key="sk-2b726a0c6b6a4554b7834df6bac0b803",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    base64_image = encode_image(image_path)

    completion = client.chat.completions.create(
        model="qwen2.5-vl-72b-instruct", 
        messages=[{"role": "user","content": [
                {"type": "text","text": f"This is a robotic arm operation scene, you need to detect {objects}. Detect all objects in the image and return their locations in the form of coordinates, don't give up any information about the details. The format of output should be like" +"{“bbox”: [x1, y1, x2, y2], “label”: the name of this object in English.} not {“bbox_2d”: [x1, y1, x2, y2], “label”: the name of this object in Chinese}"},
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


    

