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
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from torch.cuda.amp import autocast, GradScaler

# ==========================
# Load Pretrained MobileNetV3
# ==========================

weights = MobileNet_V3_Small_Weights.DEFAULT
model = mobilenet_v3_small(weights=weights)

# Freeze feature extractor
for param in model.features.parameters():
    param.requires_grad = False

# Replace classifier
model.classifier[3] = nn.Linear(1024, 2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# ==========================
# Loss & Optimizer
# ==========================

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=1e-4
)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=2
)

scaler = GradScaler()

# ==========================
# Training Settings
# ==========================

epochs = 15

best_val_acc = 0
patience = 5
counter = 0

# ==========================
# Training Loop
# ==========================

for epoch in range(epochs):

    ####################
    # TRAIN
    ####################

    model.train()

    running_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        with autocast():

            outputs = model(images)

            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()

        scaler.step(optimizer)

        scaler.update()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_acc = 100 * correct / total

    ####################
    # VALIDATION
    ####################

    model.eval()

    val_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)

            correct += (predicted == labels).sum().item()

    val_loss /= len(val_loader)
    val_acc = 100 * correct / total

    scheduler.step(val_loss)

    print("=" * 50)
    print(f"Epoch {epoch+1}/{epochs}")
    print(f"Train Loss : {train_loss:.4f}")
    print(f"Train Acc  : {train_acc:.2f}%")
    print(f"Val Loss   : {val_loss:.4f}")
    print(f"Val Acc    : {val_acc:.2f}%")

    ####################
    # SAVE BEST MODEL
    ####################

    if val_acc > best_val_acc:

        best_val_acc = val_acc

        torch.save(model.state_dict(), "best_model.pth")

        print("✅ Best model saved.")

        counter = 0

    else:

        counter += 1

    ####################
    # EARLY STOPPING
    ####################

    if counter >= patience:

        print("Early Stopping Triggered!")

        break

print("=" * 50)
print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
