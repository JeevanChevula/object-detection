YOLOv5 Object Detection on Rainy Scene
This project performs object detection on a rainy scene image using the YOLOv5 model.
Prerequisites

Python 3.x
Google Colab environment
Google Drive access

Installation

Install required packages:!pip install yolov5
!pip install pillow opencv-python


Mount Google Drive:from google.colab import drive
drive.mount('/content/drive')



Usage

Place the input image in your Google Drive (e.g., /content/drive/MyDrive/Colab Notebooks/Datasets/Yolo weights , classes and input image/A rainy scene .webp).
Run the script to:
Load the YOLOv5x model from PyTorch Hub.
Read and preprocess the input image.
Perform object detection.
Save the output image with bounding boxes to /content/drive/MyDrive/Colab Notebooks/Datasets/Yolo weights , classes and input image/rainyoutput4.jpg.



Code
import torch
import cv2
import numpy as np

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5x')

# Load and preprocess image
image_path = '/content/drive/MyDrive/Colab Notebooks/Datasets/Yolo weights , classes and input image/A rainy scene .webp'
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Perform object detection
results = model(image_rgb)

# Render results
result_image = results.render()[0]

# Save output
output_image_path = '/content/drive/MyDrive/Colab Notebooks/Datasets/Yolo weights , classes and input image/rainyoutput4.jpg'
cv2.imwrite(output_image_path, result_image)

Output
The output image (rainyoutput4.jpg) contains the input image with detected objects highlighted by bounding boxes and labels.
Notes

Ensure the input image path is correct.
The YOLOv5x model is used for high accuracy; other variants (e.g., yolov5s) can be used for faster inference.
Internet connection is required to download the model from PyTorch Hub.

