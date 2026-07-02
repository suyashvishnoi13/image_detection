"""
utils.py

Shared utility functions for training and inference.

Author: Suyash Vishnoi
Assignment: SalesCode.ai AI/ML Intern

This module centralizes:
- Device selection
- Image preprocessing
- Model creation
- Model loading

Keeping these functions in one place ensures that training and
inference always use the exact same preprocessing pipeline and model
architecture.
"""

import torch
import torch.nn as nn
from torchvision import models, transforms


# ---------------------------------------------------------
# Device Configuration
# ---------------------------------------------------------

def get_device():
    """
    Returns the available device.

    Priority:
        CUDA GPU
        Apple MPS
        CPU
    """
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


# ---------------------------------------------------------
# Image Transforms
# ---------------------------------------------------------

def get_train_transform():
    """
    Data augmentation used during training.
    """

    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.05
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def get_val_transform():
    """
    Validation / Test transform.

    IMPORTANT:
    This transform must also be used for inference.
    """

    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


# ---------------------------------------------------------
# Model
# ---------------------------------------------------------

def build_model():
    """
    Builds MobileNetV3-Small for binary classification
    using TWO output neurons.
    """

    model = models.mobilenet_v3_small(weights=None)

    in_features = model.classifier[3].in_features

    model.classifier[3] = nn.Linear(
        in_features,
        2
    )

    return model



# ---------------------------------------------------------
# Load Saved Model
# ---------------------------------------------------------

def load_model(weights_path):
    """
    Loads trained weights and returns a model
    ready for inference.

    Args:
        weights_path (str)

    Returns:
        torch.nn.Module
    """

    device = get_device()

    model = build_model()

    checkpoint = torch.load(
        weights_path,
        map_location=device
    )

    model.load_state_dict(checkpoint)

    model.to(device)

    model.eval()

    return model


# ---------------------------------------------------------
# Prediction Helper
# ---------------------------------------------------------

@torch.no_grad()
def predict_probability(model, image_tensor):
    """
    Returns probability that the image
    is a SCREEN image.
    """

    device = next(model.parameters()).device

    image_tensor = image_tensor.unsqueeze(0).to(device)

    outputs = model(image_tensor)

    probabilities = torch.softmax(outputs, dim=1)

    # Class 1 = Screen
    return probabilities[0, 1].item()