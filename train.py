import ultralytics
import shutil
from ultralytics import YOLO
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import cv2
from matplotlib import pyplot as plt
import os

ultralytics.checks()

# delete previous training artifacts if any
location = '/cnvrg/runs/detect/'
dir = 'train'
path = os.path.join(location, dir)
if os.path.exists(path):
    shutil.rmtree(path)


# Load a model
# model = YOLO('yolov8n.yaml')  # build a new model from YAML
model = YOLO('yolov8s.pt')  # load a pretrained model (recommended for training)
# model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # build from YAML and transfer weights
epochs = os.environ.get('epochs') if os.environ.get('epochs') else 10
device = os.environ.get('device') if os.environ.get('device') else 'cpu'
# Train the model
model.train(data='lpd.yaml', epochs=200, imgsz=640, batch=8, device='cpu')

# Source path 
src = '/cnvrg/runs/detect/train'
# Destination path 
dest = '/cnvrg/output/train'
# Copy the content of source to destination 
destination = shutil.copytree(src, dest) 