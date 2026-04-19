"""
train.py
Trains two ResNet50 models on CIFAR-10 with different hyperparameters/seeds
so that differential testing with DeepXplore is meaningful.

Model A: lr=0.1,  seed=42, standard augmentation
Model B: lr=0.05, seed=7,  stronger augmentation
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet50

# ───────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────

def set_seed(seed: int):
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_cifar10_loaders(batch_size: int, augmentation: str = "standard"):
    """
    Return (train_loader, test_loader) for CIFAR-10.

    augmentation:
        "standard" – horizontal flip + random crop
        "strong"   – adds color jitter and random erasing
    """
    # CIFAR-10 statistics
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2023, 0.1994, 0.2010)

    if augmentation == "standard":
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:  # "strong"
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                   saturation=0.4, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            transforms.RandomErasing(p=0.5),
        ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_set = torchvision.datasets.CIFAR10(
        root="./data", train=True,  download=True, transform=train_transform)
    test_set  = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=test_transform)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True)
    test_loader  = torch.utils.data.DataLoader(
        test_set,  batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True)

    return train_loader, test_loader


def build_resnet50_cifar10() -> nn.Module:
    """
    ResNet50 adapted for CIFAR-10 (32×32 input, 10 classes).
    The standard ResNet50 expects 224×224; we replace the first conv
    and remove the max-pool so spatial dims are preserved.
    """
    model = resnet50(weights=None)  # train from scratch

    # Replace first conv: 7×7 stride-2 → 3×3 stride-1
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1,
                            padding=1, bias=False)
    # Remove max-pool (would shrink 32→16 too aggressively)
    model.maxpool = nn.Identity()
    # Replace classifier head: 1000 → 10
    model.fc = nn.Linear(model.fc.in_features, 10)

    return model


# ───────────────────────────────────────────
# Training loop
# ───────────────────────────────────────────

def train_model(
    model_name: str,
    seed: int,
    lr: float,
    augmentation: str,
    epochs: int = 50,
    batch_size: int = 128,
    save_dir: str = "models",
):
    """Train a single model and save the best checkpoint."""
    print(f"\n{'='*60}")
    print(f"Training {model_name}  |  seed={seed}  |  lr={lr}  |  aug={augmentation}")
    print(f"{'='*60}")

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs(save_dir, exist_ok=True)
    train_loader, test_loader = get_cifar10_loaders(batch_size, augmentation)

    model = build_resnet50_cifar10().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr,
                          momentum=0.9, weight_decay=5e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0.0

    for epoch in range(1, epochs + 1):
        # ── Train ──
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total   += labels.size(0)

        scheduler.step()

        train_acc = 100.0 * correct / total

        # ── Evaluate every 5 epochs ──
        if epoch % 5 == 0 or epoch == epochs:
            model.eval()
            val_correct, val_total = 0, 0
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    _, predicted = outputs.max(1)
                    val_correct += predicted.eq(labels).sum().item()
                    val_total   += labels.size(0)

            val_acc = 100.0 * val_correct / val_total
            print(f"Epoch [{epoch:3d}/{epochs}]  "
                  f"train_loss={train_loss/total:.4f}  "
                  f"train_acc={train_acc:.2f}%  "
                  f"val_acc={val_acc:.2f}%")

            # Save best checkpoint
            if val_acc > best_acc:
                best_acc = val_acc
                ckpt_path = os.path.join(save_dir, f"{model_name}_best.pth")
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_acc": val_acc,
                    "seed": seed,
                    "lr": lr,
                    "augmentation": augmentation,
                }, ckpt_path)
                print(f"  ✓ Best model saved → {ckpt_path}  (val_acc={val_acc:.2f}%)")

    print(f"\n{model_name} training complete. Best val acc: {best_acc:.2f}%")
    return best_acc


# ───────────────────────────────────────────
# Main
# ───────────────────────────────────────────

if __name__ == "__main__":
    # ── Model A: standard settings ──
    acc_a = train_model(
        model_name="model_a",
        seed=42,
        lr=0.1,
        augmentation="standard",
        epochs=50,
    )

    # ── Model B: different seed, lower LR, stronger augmentation ──
    acc_b = train_model(
        model_name="model_b",
        seed=7,
        lr=0.05,
        augmentation="strong",
        epochs=50,
    )

    print(f"\n{'='*60}")
    print(f"Final Results:")
    print(f"  Model A best val acc: {acc_a:.2f}%")
    print(f"  Model B best val acc: {acc_b:.2f}%")
    print(f"  Checkpoints saved in: ./models/")
    print(f"{'='*60}")