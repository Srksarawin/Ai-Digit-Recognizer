"""
Trains the digit CNN on MNIST with light data augmentation so it
generalizes better to messy, hand-drawn canvas input (not just
clean pre-centered MNIST test images).

Run:
    python train.py

Produces:
    digit_cnn.pth
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from model import DigitCNN

EPOCHS = 10
BATCH_SIZE = 64
LEARNING_RATE = 1e-3

# Augmentation for training data: small rotations/shifts/zooms mimic the
# imperfections of real handwriting drawn with a mouse or touchscreen.
train_transform = transforms.Compose([
    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])

# Test data stays clean — this is our unbiased accuracy check.
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_data = datasets.MNIST(root="./data", train=True, download=True, transform=train_transform)
    test_data = datasets.MNIST(root="./data", train=False, download=True, transform=test_transform)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=1000)

    model = DigitCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Evaluate on clean test set after each epoch
        model.eval()
        correct = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                preds = model(images).argmax(dim=1)
                correct += (preds == labels).sum().item()
        accuracy = correct / len(test_data) * 100

        print(f"Epoch {epoch}/{EPOCHS} | loss: {total_loss:.2f} | test accuracy: {accuracy:.2f}%")

    torch.save(model.state_dict(), "digit_cnn.pt")
    print("\nSaved trained model to digit_cnn.pt")


if __name__ == "__main__":
    main()
