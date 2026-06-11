# Age and Gender Prediction using Lightweight CNN

## Overview

This project predicts **age group** and **gender** from face images using a lightweight Convolutional Neural Network (CNN).

The system:

* Detects all faces in an uploaded image using **MTCNN**
* Preprocesses detected faces
* Predicts:

  * Gender: Male / Female
  * Age Groups:

    * 0–18
    * 19–30
    * 31–50
    * > 50
* Displays results through a **Streamlit web application**

---

## Demo

Input Image
→ Face Detection
→ Age & Gender Prediction
→ Visualization

---

## Project Structure

```text
project/
│
├── app1.py                         # Streamlit application
├── train_gender_model_final.ipynb  # Gender model training
├── train_age_model_final.ipynb     # Age model training
├── Gender_and_Age_Prediction.ipynb # Prediction notebook
├── lightweight_cnn_64x64.pth       # Trained model
├── requirements.txt
└── README.md
```

---

## Model Architecture

### Face Detection

* MTCNN

### Classification Model

* Lightweight CNN
* Batch Normalization
* ReLU
* MaxPooling
* Dropout

Outputs:

* Gender Classification (2 classes)
* Age Classification (4 classes)

---

## Installation

Clone repository:

```bash
git clone <your-repo-link>
cd <repo-name>
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Run Application

```bash
streamlit run app1.py
```

Open browser:

```text
http://localhost:8501
```

---

## Usage

1. Upload image (.jpg / .png)
2. System detects all faces
3. Predict age and gender
4. Display prediction result

---

## Technologies

* Python
* PyTorch
* Streamlit
* OpenCV
* MTCNN
* NumPy
* Matplotlib

---

## Future Improvements

* Improve age prediction accuracy
* Add confidence score
* Deploy online
* Support real-time webcam

---


