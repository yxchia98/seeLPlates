import ultralytics
import shutil
from ultralytics import YOLO
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import cv2
from matplotlib import pyplot as plt
import json
import base64


model = YOLO('output/train/weights/best.pt')  # load a custom model

def box_label(image, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255)):
  lw = max(round(sum(image.shape) / 2 * 0.002), 1)
  p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
  cv2.rectangle(image, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)
  if label:
    tf = max(lw - 1, 1)  # font thickness
    w, h = cv2.getTextSize(label, 0, fontScale=lw / 3, thickness=tf)[0]  # text width, height
    outside = p1[1] - h >= 3
    p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
    cv2.rectangle(image, p1, p2, color, -1, cv2.LINE_AA)  # filled
    cv2.putText(image,
                label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                0,
                lw / 3,
                txt_color,
                thickness=tf,
                lineType=cv2.LINE_AA)

def plot_bboxes(image, boxes, labels=[], colors=[], score=True, conf=None):
    if labels == []:
        labels = {0: u'__background__', 1: u'plate'}
    #Define colors
    if colors == []:
        # NOTE: opencv uses the BGR format instead of RGB
        colors = [(0, 0, 255), (253, 246, 160), (40, 132, 70)]
                  
    #plot each boxes
    for box in boxes:
      #add score in label if score=True
        if score:
            label = labels[int(box[-1])+1] + " " + str(round(100 * float(box[-2]),1)) + "%"
        else:
            label = labels[int(box[-1])+1]
        #filter every box under conf threshold if conf threshold setted
        if conf:
            if box[-2] > conf:
                color = colors[int(box[-1])]
                box_label(image, box, label, color)
            else:
                color = colors[int(box[-1])]
                box_label(image, box, label, color)
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image



def predict(image_encoded):
    image_decoded = base64.b64decode(image_encoded)
    image_np = np.frombuffer(image_decoded, dtype=np.uint8)
    image = cv2.imdecode(image_np, flags=1)
    # do image predictions
    images = []
    images.append(image)
    results = model(images)
    for result in results:
        boxes = result.boxes  # Boxes object for bbox outputs
        masks = result.masks  # Masks object for segmentation masks outputs
        probs = result.probs  # Class probabilities for classification outputs
        pred = plot_bboxes(image, boxes.data, conf=0.1)
        pred_image_as_text = base64.b64encode(cv2.imencode('.jpg', pred)[1]).decode() # encode and send back image
        return pred_image_as_text
    return image_encoded
    
