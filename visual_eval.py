from camera import Camera
import matplotlib.pyplot as plt
import numpy as np
import time
from PIL import Image, ImageDraw
from VLM_demo import  show_box_cv2,show_mask_cv2,encode_image_PIL, get_world_bboxs_list,show_mask,show_box,process_visual_prompt,set_visual_prompt,predict_mask,encode_image,resize_bbox_to_original,smart_resize,get_response
import requests
from PIL import Image
import torch

from transformers import OwlViTProcessor, OwlViTForObjectDetection


processor = OwlViTProcessor.from_pretrained("./owlvit-base-patch16")
model = OwlViTForObjectDetection.from_pretrained("./owlvit-base-patch16")


#for i in range(10):
image = Image.open("tmp/images/drawer.jpg")
image = Image.fromarray(np.array(image))
image_path = f"tmp/images/drawer.jpg"
#image.save(image_path)
bbox = get_world_bboxs_list(image_path,"detect drawer handle and drawer body")
print(bbox)

texts = [["a photo of a drawer handle", "a photo of a drawer body"]]
inputs = processor(text=texts, images=image, return_tensors="pt")
outputs = model(**inputs)

# Target image sizes (height, width) to rescale box predictions [batch_size, 2]
target_sizes = torch.Tensor([image.size[::-1]])
# Convert outputs (bounding boxes and class logits) to COCO API
results = processor.post_process_object_detection(outputs=outputs, threshold=0.1, target_sizes=target_sizes)

i = 0  # Retrieve predictions for the first image for the corresponding text queries
text = texts[i]
boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]

# Print detected objects and rescaled box coordinates
for box, score, label in zip(boxes, scores, labels):
    box = [round(i, 2) for i in box.tolist()]
    print(f"Detected {text[label]} with confidence {round(score.item(), 3)} at location {box}")


bottle1 = [{'bbox': [919, 543, 1020, 622], 'label': 'bottle cap'}, {'bbox': [807, 607, 1102, 1129], 'label': 'bottle body'}]
bottle2 = [703.11, 479.18, 974.29, 993.63]

drawer1 = [{'bbox': [561, 570, 621, 706], 'label': 'drawer handle'}, {'bbox': [0, 449, 637, 1148], 'label': 'drawer body'}]
