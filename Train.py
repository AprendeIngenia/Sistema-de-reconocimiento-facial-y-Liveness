# Libraries
from ultralytics import YOLO

# Model
model = YOLO('/Modelos/yolov8l.pt')

def main():
    # Train
    model.train(data = 'CustomObjectDetect/SplitData/Dataset.yaml', epochs = 30, batch = 4, imgsz = 640)

if __name__ == '__main__':
    main()