# SalesCode.ai Assignment

## Screen vs Real Image Classification

This project classifies an input image as:

- **0 → Real Photograph**
- **1 → Photo of a Screen**

The model is based on **MobileNetV3-Small** using transfer learning.

---

## Model

Architecture:

- MobileNetV3-Small
- PyTorch
- Transfer Learning

Training Techniques:

- Data Augmentation
- Adam Optimizer
- ReduceLROnPlateau
- Mixed Precision (AMP)
- Early Stopping

---

## Dataset

Custom Dataset

```
real/
screen/
```

Total Images:

171

---

## Results

Validation Accuracy

100%

Test Accuracy

94.4%

Classification Report

| Class | Precision | Recall | F1 |
|--------|-----------|--------|----|
| Real | 1.00 | 0.88 | 0.93 |
| Screen | 0.91 | 1.00 | 0.95 |

---

## Run Prediction

```bash
python predict.py image.jpg
```

Output

```
0.203409
```

Probability

- 0 → Real
- 1 → Screen

---

## Streamlit Demo

```bash
streamlit run app.py
```

---

Author

Suyash Vishnoi