# Emotion Detection using CNN 🎭

This project implements **Facial Emotion Recognition (FER)** using Convolutional Neural Networks (CNNs).  
The model classifies faces into seven emotions: **Happy, Sad, Angry, Fear, Disgust, Neutral, Surprise**.  

---

## 🔹 Features
- Preprocessing: grayscale conversion, normalization, and data augmentation (rotation, flip, brightness/zoom).  
- Models: Base CNN and Transfer Learning (EfficientNet-B0).  
- Results:  
  - Base CNN validation accuracy: **54.45%**  
  - EfficientNet-B0 (transfer learning): **63.45%**  
- Analysis: confusion matrix showed common misclassifications (Sad ↔ Neutral, Fear ↔ Sad).  

---

## 🔹 Files in this Repo
- `Report.pdf` → Full research paper / documentation.
- `Code PDFs` → Contain the implementation code (for reference).  
- `requirements.txt` → Dependencies list.  
- `README.md` → Project overview (this file).  

---

## 🔹 How to Run (if converted to .py files)
```bash
pip install -r requirements.txt
python train.py   # or the main training file
