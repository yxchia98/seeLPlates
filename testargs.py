import argparse

# Takes in CLI arguments
argParser = argparse.ArgumentParser(description="set parameters for YOLOv8 model training")
argParser.add_argument("-e", "--epochs", default=50, type=int, help="number of epochs to be used, default to 50")
argParser.add_argument("-b", "--batch", default="16", type=int, help="specify batch size, default to 16")
argParser.add_argument("-m", "--model", default="yolov8n.pt", help="YOLOv8 model to be used: yolov8n.pt, yolov8s.pt, yolov8m.pt. defaults to yolov8n.pt")
args = argParser.parse_args()
print(args.epochs, args.epochs.__class__)
print(args.batch, args.batch.__class__)
print(args.model, args.model.__class__)
