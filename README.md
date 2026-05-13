# AI-Generated Image Detection using ResNet50

## Overview
This project detects whether an uploaded image is real or AI-generated using a ResNet50 transfer learning model. It performs binary image classification and displays the prediction result with a confidence score using a Streamlit web interface.

## Features
- Upload JPG, JPEG, or PNG images
- Classify image as Real or AI-Generated
- Confidence score display
- ResNet50 transfer learning model
- Streamlit web interface
- Around 89% validation accuracy on the test dataset

## Tech Stack
- Python
- TensorFlow / Keras
- ResNet50
- Streamlit
- NumPy
- Pillow
- Matplotlib

## Model
The model uses ResNet50 pretrained on ImageNet as a frozen feature extractor. A custom classifier head is trained on real and AI-generated images for binary classification.

## Project Structure
```text
ai-image-detection-resnet50/
│
├── app.py
├── train_model.py
├── ai_image_resnet50_model.keras
├── requirements.txt
├── training_accuracy.png
├── README.md
└── .gitignore

