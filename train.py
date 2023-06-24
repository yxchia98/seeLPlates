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
import subprocess
import torch



ultralytics.checks()

# delete previous training artifacts if any
location = '/cnvrg/runs/detect/'
dir = 'train'
path = os.path.join(location, dir)
if os.path.exists(path):
    shutil.rmtree(path)

# input_model = os.environ.get('model') + '.pt' if os.environ.get('model') else 'yolov8n.pt'
input_model = 'yolov8m.pt'

# Load a model
# model = YOLO('yolov8n.yaml')  # build a new model from YAML
model = YOLO(input_model)  # load a pretrained model (recommended for training)
# model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # build from YAML and transfer weights
# epochs = os.environ.get('epochs') if os.environ.get('epochs') else 100
# device = os.environ.get('device') if os.environ.get('device') else 'cpu'
epochs = 100
device = '0' if torch.cuda.is_available() else 'cpu'
batch_size = 8
# Train the model
model.train(data='lpd.yaml', epochs=epochs, imgsz=640, batch=batch_size, device=device)

# Delete previous output artifacts
location = '/cnvrg/output/'
dir = 'train'
path = os.path.join(location, dir)
if os.path.exists(path):
    shutil.rmtree(path)
    
# Source path 
src = '/cnvrg/runs/detect/train'
# Destination path 
dest = '/cnvrg/output/train'
# Copy the content of source to destination 
destination = shutil.copytree(src, dest) 


# Commit to Git
subprocess.run(["pwd"])
subprocess.run(["git", "status"])
subprocess.run(["git", "add", "."])
subprocess.run(["git", "commit", "-m", "'experiment commit'"])
subprocess.run(["git", "push", "origin", "main"])