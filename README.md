# TB & Pneumonia Diagnostic System

An AI-powered healthcare application for detecting **Tuberculosis (TB)** and **Pneumonia** using **Chest X-ray images** and **clinical symptoms**. The system combines **Convolutional Neural Networks (CNN)**, **MobileNet (Transfer Learning)**, and **Random Forest** to improve prediction accuracy through a multimodal approach.

---

## 📌 Overview

Tuberculosis and Pneumonia are serious respiratory diseases that require early diagnosis for effective treatment. Manual diagnosis through chest X-rays can be time-consuming and depends on experienced radiologists.

This project leverages Artificial Intelligence to assist healthcare professionals by analyzing:

- Chest X-ray images using CNN and MobileNet
- Clinical symptoms using Random Forest
- Combining predictions to provide a final diagnosis

The system classifies patients into one of the following categories:

- Tuberculosis (TB)
- Pneumonia
- Normal

---

## 🎯 Objectives

- Detect Tuberculosis from Chest X-rays.
- Detect Pneumonia from Chest X-rays.
- Analyze patient symptoms using Machine Learning.
- Improve prediction accuracy through multimodal learning.
- Reduce diagnosis time.
- Provide decision support for healthcare professionals.

---

## 🛠 Technologies Used

| Category | Technology |
|----------|------------|
| Programming Language | Python |
| Deep Learning | TensorFlow, Keras |
| CNN Model | Custom CNN |
| Transfer Learning | MobileNet |
| Machine Learning | Random Forest |
| Image Processing | OpenCV |
| Data Analysis | NumPy, Pandas |
| Visualization | Matplotlib |
| Web Application | Streamlit |
| Model Saving | Joblib, H5 |

---

## 📂 Dataset

The project uses two datasets:

### Chest X-ray Dataset

Contains images belonging to:

- Tuberculosis
- Pneumonia
- Normal

### Clinical Dataset

Contains patient information such as:

- Age
- Gender
- Fever
- Cough
- Chest Pain
- Difficulty Breathing
- Fatigue
- Weight Loss
- Night Sweats

---

## 🏗 Project Architecture

```
                     Patient
                        │
          ┌─────────────┴─────────────┐
          │                           │
   Chest X-ray Image         Clinical Symptoms
          │                           │
  Image Preprocessing        Data Preprocessing
          │                           │
     CNN + MobileNet         Random Forest
          │                           │
          └──────────┬────────────────┘
                     │
          Final Prediction Module
                     │
      TB / Pneumonia / Normal
```

---

## ⚙ Working Process

### Step 1: Data Collection

Collect Chest X-ray images and patient symptom data.

### Step 2: Data Preprocessing

#### Image Preprocessing

- Resize images
- Normalize pixel values
- Data augmentation
- Image enhancement

#### Clinical Data Preprocessing

- Handle missing values
- Encode categorical features
- Feature selection

---

### Step 3: CNN Model

The CNN extracts important image features such as:

- Lung texture
- Infection regions
- Opacity
- Abnormal patterns

CNN Architecture:

- Convolution Layer
- ReLU Activation
- Max Pooling
- Flatten
- Dense Layer
- Softmax Output

---

### Step 4: MobileNet Model

MobileNet is used through Transfer Learning.

Advantages:

- Lightweight architecture
- Faster training
- High accuracy
- Suitable for deployment

The final classification layer is customized to predict:

- TB
- Pneumonia
- Normal

---

### Step 5: Random Forest

Clinical symptoms are analyzed using the Random Forest algorithm.

Example Features:

| Feature | Example |
|----------|----------|
| Fever | Yes |
| Cough | Yes |
| Weight Loss | Yes |
| Chest Pain | No |
| Age | 42 |

Advantages:

- Handles multiple features efficiently
- Reduces overfitting
- High classification accuracy
- Easy to interpret

---

### Step 6: Final Prediction

Predictions from:

- CNN
- MobileNet
- Random Forest

are combined to generate the final diagnosis.

Example Output:

```
CNN Prediction         : Tuberculosis (94%)

MobileNet Prediction   : Tuberculosis (96%)

Random Forest          : Tuberculosis (91%)

Final Prediction       : Tuberculosis
Confidence             : 95%
```

---

## 📁 Project Structure

```
TB-Pneumonia-Diagnostic-System/
│
├── dataset/
│   ├── train/
│   ├── validation/
│   └── test/
│
├── models/
│   ├── cnn_model.h5
│   ├── mobilenet_model.h5
│   └── random_forest.pkl
│
├── notebooks/
│   ├── cnn_training.ipynb
│   ├── mobilenet_training.ipynb
│   └── random_forest_training.ipynb
│
├── app.py
├── requirements.txt
├── README.md
├── images/
└── utils/
```

---

## 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/your-username/TB-Pneumonia-Diagnostic-System.git
```

Move into the project directory:

```bash
cd TB-Pneumonia-Diagnostic-System
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

---

## 📦 Requirements

```
tensorflow
keras
numpy
pandas
opencv-python
matplotlib
scikit-learn
streamlit
Pillow
joblib
```

---

## ✨ Features

- Detects Tuberculosis
- Detects Pneumonia
- Identifies Normal Chest X-rays
- Clinical symptom analysis
- CNN-based feature extraction
- MobileNet Transfer Learning
- Random Forest classification
- User-friendly Streamlit interface
- Fast and accurate prediction
- Multimodal disease diagnosis

---

## 📈 Future Enhancements

- Add COVID-19 detection.
- Support Lung Cancer detection.
- Improve model accuracy with larger datasets.
- Cloud deployment using AWS.
- Mobile application integration.
- Explainable AI (XAI) for prediction visualization.
- Integration with hospital information systems.

---

## 📄 License

This project is developed for educational and research purposes. You are free to use and modify it for academic work with proper attribution.
