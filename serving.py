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
import torch


model = YOLO('output/train/weights/best.pt')  # load a custom model
device = '0' if torch.cuda.is_available() else 'cpu'

# def box_label(image, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255)):
#   lw = max(round(sum(image.shape) / 2 * 0.00001), 2)
#   p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
#   cv2.rectangle(image, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)
#   if label:
#     tf = max(lw - 1, 1)  # font thickness
#     w, h = cv2.getTextSize(label, 0, fontScale=lw / 3, thickness=tf)[0]  # text width, height
#     outside = p1[1] - h >= 3
#     p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
#     cv2.rectangle(image, p1, p2, color, -1, cv2.LINE_AA)  # filled
#     cv2.putText(image,
#                 label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
#                 0,
#                 lw / 3,
#                 txt_color,
#                 thickness=tf,
#                 lineType=cv2.LINE_AA)

# def plot_bboxes(image, boxes, labels=[], colors=[], score=True, conf=None):
#     if labels == []:
#         labels = {0: u'__background__', 1: u'Plate'}
#     #Define colors
#     if colors == []:
#         # NOTE: opencv uses the BGR format instead of RGB
#         colors = [(0, 0, 255), (253, 246, 160), (40, 132, 70)]
                  
#     #plot each boxes
#     for box in boxes:
#       #add score in label if score=True
#         if score:
#             label = labels[int(box[-1])+1] + " " + str(round(100 * float(box[-2]),1)) + "%"
#         else:
#             label = labels[int(box[-1])+1]
#         #filter every box under conf threshold if conf threshold setted
#         if conf:
#             if box[-2] > conf:
#                 color = colors[int(box[-1])]
#                 box_label(image, box, label, color)
#             else:
#                 color = colors[int(box[-1])]
#                 box_label(image, box, label, color)
    
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     return image

class PredictionResponse:
    def __init__(self):
        self.image = ""
        self.cropped_results = []

def box_label(image, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255)):
  lw = max(round(sum(image.shape) / 2 * 0.00001), 2)
  p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
  
  # extract cartesian coordinates, crop image, and make a copy of the array
  x1, y1, x2, y2 = p1[0], p1[1], p2[0], p2[1]
  roi = image[y1:y2, x1:x2].copy()

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
  return cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

def plot_bboxes(image, boxes, labels=[], colors=[], score=True, conf=None):

    res = PredictionResponse()


    if labels == []:
        labels = {0: u'__background__', 1: u'Plate'}
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
            print(box[-2])
            if box[-2] > conf:
                color = colors[int(box[-1])]
                roi = box_label(image, box, label, color)
                res.cropped_results.append(roi)
            # else:
            #     color = colors[int(box[-1])]
            #     box_label(image, box, label, color)
        
    
    res.image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return res



def predict(image_encoded):
    image_decoded = base64.b64decode(image_encoded)
    image_np = np.frombuffer(image_decoded, dtype=np.uint8)
    image = cv2.imdecode(image_np, flags=1)

    res = {
    'image': image_encoded,
    'cropped_images': []
    }

    # do image predictions
    images = []
    images.append(image)
    results = model.predict(images, device=device)
    for result in results:
        boxes = result.boxes  # Boxes object for bbox outputs
        masks = result.masks  # Masks object for segmentation masks outputs
        probs = result.probs  # Class probabilities for classification outputs
        pred = plot_bboxes(image, boxes.data, conf=0.1)
        pred_image = base64.b64encode(cv2.imencode('.jpg', pred.image)[1]).decode() # encode and send back image
        cropped_images = []
        for i in pred.cropped_results:
            cropped_result_as_text = base64.b64encode(cv2.imencode('.jpg', i)[1]).decode()
            cropped_images.append(cropped_result_as_text)
        res = {
            'image': pred_image,
            'cropped_images': cropped_images
        }
    return res
    
