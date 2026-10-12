# Automated Skin Cancer Detection System

## Overview
This project is a deep learning-based web application that classifies skin lesion images using a Convolutional Neural Network (CNN). It uses TensorFlow/Keras for model development and Flask to provide a web interface for image upload and prediction.

## Technologies Used
- Python
- TensorFlow and Keras
- Flask
- HTML and CSS
- NumPy
- HAM10000 skin lesion image dataset

## Features
- Upload skin lesion images through a web interface
- Preprocess images before prediction
- Predict image classes using a trained CNN model
- Display prediction results through a Flask application
- Visualize model performance using accuracy, loss, ROC curve, and confusion matrix plots

## Project Structure
```text
SkinCancerAI/
├── app.py
├── predict.py
├── train_model.py
├── skin_cancer_model.h5
├── dataset/
├── static/
├── templates/
├── uploads/
├── README.md
└── requirements.txt
```

## Installation and Setup

### 1. Clone the repository
```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd automated-skin-cancer-detection
```

### 2. Create a virtual environment
```bash
python -m venv venv
```

### 3. Activate the environment
On Windows PowerShell:
```powershell
.\venv\Scripts\Activate.ps1
```

### 4. Install dependencies
```bash
pip install -r requirements.txt
```

### 5. Run the application
```bash
python app.py
```

Open your browser and visit:
`http://127.0.0.1:5000`

Make sure the trained model is present at the file path expected by the application.

## Model Evaluation
This project includes plots for accuracy, loss, ROC curve, and confusion matrix. Refer to the generated evaluation results to understand the model's performance.

## Future Enhancements
- Improve model performance and generalization
- Add precision, recall, and F1-score reporting
- Integrate Grad-CAM for model explainability
- Deploy the application to a cloud platform

## Disclaimer
This project is intended for educational purposes only. It is not a medical diagnostic tool. Predictions may be inaccurate and must not be used as a substitute for professional medical advice.
