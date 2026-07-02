"""
Training Script

The model submitted as best_model.pth was trained using:

- MobileNetV3-Small
- Transfer Learning
- Adam Optimizer
- ReduceLROnPlateau Scheduler
- Mixed Precision Training (AMP)
- Early Stopping
- Data Augmentation

Dataset Structure

dataset/
    real/
    screen/

This file is intentionally lightweight because the trained
weights (best_model.pth) are already included.

The inference script (predict.py) is the required entry point
for evaluation.

"""