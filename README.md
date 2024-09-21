# Rock-Paper-Scissors Detection with YOLOv8

This project implements a real-time rock-paper-scissors gesture recognition system using the YOLOv8 model. The model is trained on a dataset from Roboflow and can recognize gestures through a webcam feed.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Dataset](#dataset)
- [Training the Model](#training-the-model)
- [Real-time Detection](#real-time-detection)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Prerequisites

Ensure you have the following installed:
- Python 3.x
- pip

## Installation

1. Clone the repository:

   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Install required libraries:

   ```bash
   pip install ultralytics roboflow opencv-python
   ```

## Dataset

This project uses the rock-paper-scissors dataset available on Roboflow. You need an API key to access the dataset. Make sure to replace the placeholder with your own API key.

## Training the Model

To train the YOLOv8 model, run the following code:

```python
from ultralytics import YOLO
from roboflow import Roboflow

# Load the YOLO model
model = YOLO('yolov8n.pt')

# Download the dataset from Roboflow
rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("roboflow-58fyf").project("rock-paper-scissors-sxsw")
version = project.version(14)
dataset = version.download("yolov8")

# Train the model
!yolo task=detect mode=train model=yolov8n.pt data=/content/rock-paper-scissors-14/data.yaml epochs=10 imgsz=640 plots=True
```

After training, a confusion matrix will be displayed.

## Real-time Detection

To set up a webcam for real-time gesture detection, use the following code:

```python
import cv2

# Load the best trained model
model = YOLO("best.pt")
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    result = model.predict(source=frame, conf=0.6)
    frame = result[0].plot()  # Ensure this returns a frame-compatible image

    cv2.imshow("frame", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

## Usage

1. Run the training script to train the model.
2. Execute the real-time detection code to start recognizing gestures using your webcam.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any bugs or enhancements.


## Author
Rana Ahmed
LinkedIn: linkedin.com/in/rana-ahmed-11
