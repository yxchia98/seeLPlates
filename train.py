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
import argparse


# Takes in CLI arguments
argParser = argparse.ArgumentParser(description="set parameters for YOLOv8 model training")
argParser.add_argument("-e", "--epochs", default=50, type=int, help="number of epochs to be used, default to 50")
argParser.add_argument("-b", "--batch", default="16", type=int, help="specify batch size, default to 16")
argParser.add_argument("-m", "--model", default="yolov8n.pt", help="YOLOv8 model to be used: yolov8n.pt, yolov8s.pt, yolov8m.pt. defaults to yolov8n.pt")
args = argParser.parse_args()

ultralytics.checks()

# delete previous training artifacts if any
location = '/cnvrg/runs/detect/'
dir = 'train'
path = os.path.join(location, dir)
if os.path.exists(path):
    shutil.rmtree(path)

# input_model = os.environ.get('model') + '.pt' if os.environ.get('model') else 'yolov8n.pt'
# input_model = 'yolov8s.pt'
input_model = args.model
batch_size = args.batch
epochs = args.epochs

# Load a model
# model = YOLO('yolov8n.yaml')  # build a new model from YAML
model = YOLO(input_model)  # load a pretrained model (recommended for training)
# model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # build from YAML and transfer weights
# epochs = os.environ.get('epochs') if os.environ.get('epochs') else 100
# device = os.environ.get('device') if os.environ.get('device') else 'cpu'
device = '0' if torch.cuda.is_available() else 'cpu'
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

# Delete previous output artifacts from yolov8
location = '/cnvrg/runs/detect/'
dir = 'train'
path = os.path.join(location, dir)
if os.path.exists(path):
    shutil.rmtree(path)


# Commit to Git
subprocess.run(["pwd"])
subprocess.run(["git", "status"])
subprocess.run(["git", "add", "."])
subprocess.run(["git", "commit", "-m", "'experiment commit'"])
subprocess.run(["git", "push", "origin", "main"])