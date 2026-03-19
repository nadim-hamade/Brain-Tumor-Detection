import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models

DATA_DIR    = Path("../data")
MODELS_DIR  = Path("../models")
RESULTS_DIR = Path("../results")

CLASSES     = ['glioma', 'meningioma', 'notumor', 'pituitary']
NUM_CLASSES = len(CLASSES)
IMG_SIZE    = 224
BATCH_SIZE  = 32
EPOCHS      = 16
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Dataset
class BrainTumorDataset(Dataset):
    def __init__(self, data_dir, split, transform=None):
        self.data_dir  = Path(data_dir) / split
        self.transform = transform
        self.samples   = []

        for label, cls in enumerate(CLASSES):
            class_path = self.data_dir / cls
            for img_name in os.listdir(class_path):
                self.samples.append((class_path / img_name, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2,
                           saturation=0.2, hue=0.2),
    transforms.RandomAffine(degrees=0, shear=10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def get_dataloaders():
    train_dataset = BrainTumorDataset(DATA_DIR, "training", train_transform)
    test_dataset  = BrainTumorDataset(DATA_DIR, "testing",  test_transform)

    train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                               shuffle=True,  num_workers=2)
    test_loader   = DataLoader(test_dataset,  batch_size=BATCH_SIZE,
                               shuffle=False, num_workers=2)

    print(f"Training samples : {len(train_dataset)}")
    print(f"Testing samples  : {len(test_dataset)}")

    return train_loader, test_loader, train_dataset

def build_model():
    model = models.resnet50(weights="IMAGENET1K_V1")

    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze layer4 for fine-tuning
    for param in model.layer4.parameters():
        param.requires_grad = True

    # Replace final classifier
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(model.fc.in_features, 256),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(256, NUM_CLASSES)
    )

    model = model.to(DEVICE)
    print("ResNet50 loaded with pretrained ImageNet weights")
    print(f"   Output classes : {NUM_CLASSES}")
    print(f"   Device         : {DEVICE}")
    return model

# Training
def train_model(model, train_loader, test_loader, train_dataset):
    # Auto-compute class weights
    train_labels  = [label for _, label in train_dataset]
    class_weights = compute_class_weight('balanced',
                                          classes=np.unique(train_labels),
                                          y=train_labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)
    print(f"   Class weights  : {class_weights.cpu().numpy().round(3)}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Two learning rates — layer4 slower, classifier faster
    optimizer = optim.Adam([
        {'params': model.layer4.parameters(), 'lr': 1e-5},
        {'params': model.fc.parameters(),     'lr': 1e-4}
    ], weight_decay=1e-4)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    history  = {'train_loss': [], 'train_acc': [],
                'test_loss':  [], 'test_acc':  []}
    best_acc = 0.0

    for epoch in range(EPOCHS):
        # Train
        model.train()
        train_loss, train_correct = 0.0, 0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss    += loss.item() * images.size(0)
            train_correct += (outputs.argmax(1) == labels).sum().item()

        model.eval()
        test_loss, test_correct = 0.0, 0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs       = model(images)
                loss          = criterion(outputs, labels)
                test_loss    += loss.item() * images.size(0)
                test_correct += (outputs.argmax(1) == labels).sum().item()

        train_loss = train_loss / len(train_loader.dataset)
        train_acc  = train_correct / len(train_loader.dataset)
        test_loss  = test_loss  / len(test_loader.dataset)
        test_acc   = test_correct / len(test_loader.dataset)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)

        print(f"Epoch [{epoch+1:02d}/{EPOCHS}] "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Test Loss: {test_loss:.4f} Acc: {test_acc:.4f}")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), MODELS_DIR / "best_model.pth")
            print(f"    Best model saved! Acc: {best_acc:.4f}")

        scheduler.step()

    print(f"\n Best Test Accuracy: {best_acc:.4f}")
    return history

# Evaluation
def evaluate_model(model, test_loader):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images  = images.to(DEVICE)
            outputs = model(images)
            preds   = outputs.argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    print("\n Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=CLASSES))

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=150,
                bbox_inches='tight')
    plt.show()
    print(" Confusion matrix saved!")

def plot_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history['train_acc'], label='Train Accuracy', color='steelblue')
    axes[0].plot(history['test_acc'],  label='Test Accuracy',  color='coral')
    axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(history['train_loss'], label='Train Loss', color='steelblue')
    axes[1].plot(history['test_loss'],  label='Test Loss',  color='coral')
    axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "training_history.png", dpi=150,
                bbox_inches='tight')
    plt.show()
    print(" Training history saved!")

if __name__ == "__main__":
    print("Brain Tumor Detection — Training")
    print("=" * 50)

    MODELS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    train_loader, test_loader, train_dataset = get_dataloaders()
    model   = build_model()
    history = train_model(model, train_loader, test_loader, train_dataset)

    evaluate_model(model, test_loader)
    plot_history(history)

    print("\n Training complete!")
    print(f"   Model saved   → models/best_model.pth")
    print(f"   Results saved → results/")