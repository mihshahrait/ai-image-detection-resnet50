# AI-Generated Image Detection using ResNet50

## 📌 Overview

AI-Generated Image Detection is a computer vision project that classifies an uploaded image as either **Real** or **AI-Generated** using a ResNet50-based transfer learning model.

The project uses a pretrained **ResNet50 CNN** as a feature extractor and adds a custom binary classification head for detecting AI-generated images. A Streamlit web interface allows users to upload an image and receive a prediction with a confidence score.

This project was built as a demo-level machine learning application for image authenticity detection.

---

## 🚀 Live Demo

Live App: _Add your Streamlit deployment link here_

Example:

```text
https://your-app-name.streamlit.app
```

---

## ✨ Features

- Upload image files in JPG, JPEG, or PNG format
- Classify images as:
  - Real Image
  - AI-Generated Image
- Display prediction confidence score
- Clean Streamlit-based web interface
- ResNet50 transfer learning model
- Custom HTML/CSS styling inside Streamlit
- Training accuracy graph included
- Demo-level validation accuracy of around **89%**

---

## 🧠 Model Used

The model is built using **ResNet50**, a deep convolutional neural network pretrained on ImageNet.

### Model Approach

- ResNet50 base model is loaded with pretrained ImageNet weights
- Top classification layer is removed
- ResNet50 layers are frozen during training
- A custom classifier head is added
- Binary classification is performed using a sigmoid output layer

### Classification Labels

```text
0 → AI-Generated Image
1 → Real Image
```

---

## 📊 Training Results

The model was trained on a subset of real and AI-generated images.

| Metric | Result |
|---|---:|
| Final Training Accuracy | ~85.66% |
| Final Validation Accuracy | ~88.90% |
| Best Validation Accuracy | ~89.00% |

The model used transfer learning with only the custom classifier head being trainable, while the ResNet50 base remained frozen.

---

## 🛠️ Tech Stack

| Category | Tools / Libraries |
|---|---|
| Programming Language | Python |
| Deep Learning | TensorFlow, Keras |
| CNN Architecture | ResNet50 |
| Web App | Streamlit |
| Image Processing | Pillow |
| Numerical Computing | NumPy |
| Visualization | Matplotlib |
| Version Control | Git, GitHub |

---

## 📁 Project Structure

```text
ai-image-detection-resnet50/
│
├── app.py                         # Streamlit web application
├── train_model.py                 # Model training script
├── ai_image_resnet50_model.keras  # Trained ResNet50 model
├── training_accuracy.png          # Accuracy graph from training
├── requirements.txt               # Python dependencies
├── README.md                      # Project documentation
└── .gitignore                     # Ignored files and folders
```

---

## ⚙️ Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/mihshahrait/ai-image-detection-resnet50.git
cd ai-image-detection-resnet50
```

### 2. Create a Virtual Environment

#### Windows Git Bash

```bash
python -m venv venv
source venv/Scripts/activate
```

#### Windows PowerShell

```powershell
python -m venv venv
.\venv\Scripts\activate
```

#### Linux / Mac

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ How to Run the App

Run the Streamlit application:

```bash
python -m streamlit run app.py
```

After running, open the local URL shown in the terminal:

```text
http://localhost:8501
```

---

## 🏋️ How to Train the Model

The dataset folder should follow this structure:

```text
dataset/
│
├── train/
│   ├── ai_generated/
│   └── real/
│
└── val/
    ├── ai_generated/
    └── real/
```

Then run:

```bash
python train_model.py
```

After training, the model will be saved as:

```text
ai_image_resnet50_model.keras
```

---

## 📂 Dataset

The model was trained using a real vs AI-generated image dataset.

The dataset is not included in this repository due to file size limitations.

Expected dataset format:

```text
Real images          → dataset/train/real
AI-generated images  → dataset/train/ai_generated
Validation real      → dataset/val/real
Validation AI images → dataset/val/ai_generated
```

---

## 🖥️ Application Workflow

```text
User uploads image
        ↓
Image is resized to 224x224
        ↓
Image is passed to ResNet50 model
        ↓
Model predicts Real or AI-Generated
        ↓
Confidence score is displayed
```

---

## 📸 Screenshots

Add screenshots here after deployment:

```markdown
![Home Page](screenshots/home.png)
![Prediction Result](screenshots/result.png)
```

---

## ✅ Example Output

```text
Prediction: AI-Generated Image
Confidence: 97.42%
```

or

```text
Prediction: Real Image
Confidence: 91.85%
```

---

## 🔮 Future Scope

- Train on a larger and more diverse dataset
- Improve accuracy on real-world phone images
- Add Grad-CAM visualization to show important image regions
- Add support for WEBP images
- Add batch image prediction
- Add downloadable prediction reports
- Deploy a lighter MobileNetV2 version for faster cloud inference
- Fine-tune selected ResNet50 layers for better performance

---

## ⚠️ Limitations

This is a demo-level academic project.

The model may give incorrect predictions for:

- High-resolution phone images
- Screenshots
- Heavily compressed images
- Edited or filtered images
- Images from sources not represented in the training dataset
- AI images generated by newer models not present in the dataset

The confidence score should not be treated as forensic proof.

---

## 📌 Disclaimer

This tool is intended for educational and demonstration purposes only. It should not be used as the sole method for verifying whether an image is real or AI-generated.

---

## 👤 Author

**Mihir Shah**

GitHub: [mihshahrait](https://github.com/mihshahrait)
