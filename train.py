import ultralytics
import shutil
from ultralytics import YOLO
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import cv2
from matplotlib import pyplot as plt

ultralytics.checks()

# Load a model
# model = YOLO('yolov8n.yaml')  # build a new model from YAML
model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
# model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # build from YAML and transfer weights

# Train the model
model.train(data='lpd.yaml', epochs=10, imgsz=640, device='cpu')

# Source path 
src = '/cnvrg/runs/detect/train'
# Destination path 
dest = '/cnvrg/output/train'
# Copy the content of source to destination 
destination = shutil.copytree(src, dest) 