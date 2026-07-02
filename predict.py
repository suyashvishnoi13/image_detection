"""
predict.py

SalesCode.ai Assignment

Usage:
    python predict.py image.jpg

Output:
    A single probability between 0 and 1.

Interpretation:
    0 -> REAL PHOTO
    1 -> PHOTO OF A SCREEN
"""

import sys
from pathlib import Path

from PIL import Image

from utils import (
    load_model,
    get_val_transform,
    predict_probability,
)

# Model file
MODEL_PATH = "best_model.pth"


def main():

    # Check command line arguments
    if len(sys.argv) != 2:
        print("Usage: python predict.py <image_path>")
        sys.exit(1)

    image_path = Path(sys.argv[1])

    # Check if image exists
    if not image_path.exists():
        print(f"Error: {image_path} not found.")
        sys.exit(1)

    try:
        # Load image
        image = Image.open(image_path).convert("RGB")

        # Preprocess
        transform = get_val_transform()
        image_tensor = transform(image)

        # Load model
        model = load_model(MODEL_PATH)

        # Predict
        probability = predict_probability(model, image_tensor)

        # IMPORTANT:
        # Print ONLY probability for valid input.
        print(f"{probability:.6f}")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()