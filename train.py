import ultralytics
import shutil 

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