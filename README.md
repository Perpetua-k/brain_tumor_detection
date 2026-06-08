🧠 Brain Tumor Detection Using Deep Learning (CNN)
This project is a medical image classification system that leverages a Convolutional Neural Network (CNN) to detect the presence of brain tumors from MRI scans. The solution is deployed using Streamlit, enabling an interactive web interface where users can upload MRI images and receive real-time predictions.

The system classifies images into two categories:

Tumor
No Tumor

This project demonstrates practical application of Deep Learning in Computer Vision for healthcare diagnostics support.

🎯 Objectives
Build an end-to-end deep learning pipeline for medical image classification
Train a CNN model using MRI brain scan datasets
Deploy a user-friendly web application using Streamlit
Enable real-time inference on unseen MRI images

🧠 Model Architecture

A Convolutional Neural Network (CNN) was designed with the following structure:

Convolutional Layers (feature extraction)
MaxPooling Layers (dimensionality reduction)
Flatten Layer
Fully Connected Dense Layers
Sigmoid Activation for binary classification
Input Specifications:
Image Size: 150 × 150 pixels
Color Mode: RGB
Output: Binary classification (Tumor / No Tumor)


🛠️ Tech Stack
Python
TensorFlow / Keras
Convolutional Neural Networks (CNN)
Streamlit (Web Deployment)
NumPy
Pillow (PIL)
OpenCV (optional preprocessing)

brain_tumor_detection/
│
├── app.py                    # Streamlit web application
├── train_model.py           # CNN model training script
├── requirements.txt         # Dependencies
├── .gitignore               # Ignored files configuration
│
├── dataset/                 # MRI dataset (excluded from GitHub)
│   ├── yes/                 # Tumor images
│   ├── no/                  # Non-tumor images
│
└── brain_tumor_model.h5     # Trained model (excluded due to size)


⚙️ Model Training Pipeline
Dataset loading and preprocessing using ImageDataGenerator
Image normalization (rescaling pixel values)
Train/validation split (80/20)
CNN model training using binary cross-entropy loss
Model evaluation on validation dataset
Model persistence as .h5 file
