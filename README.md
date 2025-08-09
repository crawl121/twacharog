# TwachaRog – AI-Based Skin Disease Detection System

**Twacha Rog** is an AI-powered skin disease detection system designed to assist users in identifying skin diseases through image classification. The project integrates a deep learning model trained onv 
dermatological image datasets with a web-based user interface built using TypeScript.

## Features
**AI-Powered Detection:** Uses a pre-trained MobileNetV2 model fine-tuned on a custom skin disease dataset.
**Image Upload Support:** Users can upload images of affected skin for instant analysis.
**Flask Backend API:** Serves the trained model and handles inference.
**TypeScript Frontend:** Clean, interactive UI with real-time predictions.
**High Accuracy:** Model trained to achieve over 97% training accuracy and 93% validation accuracy.

## Tech Stack

| Component         | Technology                    |
|-------------------|-------------------------------|
| Model             | TensorFlow / MobileNetV2      |
| Server            | Flask (Python)                |
| Frontend          | HTML, CSS, JavaScript         |
| Data Augmentation | TensorFlow ImageDataGenerator |
