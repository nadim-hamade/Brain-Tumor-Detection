import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from pathlib import Path
import random

CLASSES     = ['glioma', 'meningioma', 'notumor', 'pituitary']
IMG_SIZE    = 224
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODELS_DIR  = Path("../models")
RESULTS_DIR = Path("../results")

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def load_model():
    model = models.resnet50(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(model.fc.in_features, 256),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(256, len(CLASSES))
    )
    model.load_state_dict(torch.load(MODELS_DIR / "best_model.pth",
                                     map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    print("ResNet50 model loaded successfully!")
    return model

# Predict Single Image
def predict_image(image_path, model):
    img        = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs       = model(img_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        class_idx     = probabilities.argmax().item()
        confidence    = probabilities[class_idx].item()

    return {
        "class"        : CLASSES[class_idx],
        "confidence"   : confidence,
        "probabilities": {cls: probabilities[i].item()
                          for i, cls in enumerate(CLASSES)}
    }

# Visualize Prediction
def visualize_prediction(image_path, model, save_path=None):
    result = predict_image(image_path, model)
    img    = Image.open(image_path).convert("RGB")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Brain Tumor Detection", fontsize=16, fontweight='bold')

    axes[0].imshow(img)
    axes[0].set_title(f"Predicted: {result['class'].capitalize()}\n"
                      f"Confidence: {result['confidence']:.2%}")
    axes[0].axis('off')

    classes = list(result['probabilities'].keys())
    probs   = list(result['probabilities'].values())
    colors  = ['coral' if cls == result['class']
               else 'steelblue' for cls in classes]

    axes[1].barh(classes, probs, color=colors)
    axes[1].set_xlim(0, 1)
    axes[1].set_xlabel('Probability')
    axes[1].set_title('Class Probabilities')

    for i, prob in enumerate(probs):
        axes[1].text(prob + 0.01, i, f'{prob:.2%}', va='center')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Prediction saved  {save_path}")

    plt.show()
    return result

# Batch Prediction
def predict_batch(folder_path, model):
    folder_path = Path(folder_path)
    results     = []

    for img_name in os.listdir(folder_path):
        if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = folder_path / img_name
            result   = predict_image(img_path, model)
            results.append({
                "image"     : img_name,
                "predicted" : result['class'],
                "confidence": f"{result['confidence']:.2%}"
            })
            print(f"  {img_name:<30} → {result['class']:<12} "
                  f"({result['confidence']:.2%})")

    return results

if __name__ == "__main__":
    model = load_model()

    # Test on a random image
    cls          = random.choice(['glioma', 'meningioma', 'notumor', 'pituitary'])
    sample_dir   = Path(f"../data/testing/{cls}")
    sample_image = sample_dir / random.choice(os.listdir(sample_dir))

    print(f"\nPredicting: {sample_image.name}")
    RESULTS_DIR.mkdir(exist_ok=True)
    result = visualize_prediction(sample_image, model,
                                  save_path=RESULTS_DIR / "sample_prediction.png")

    print(f"\n📋 Result:")
    print(f"   Class      : {result['class'].capitalize()}")
    print(f"   Confidence : {result['confidence']:.2%}")
    print(f"\n   All probabilities:")
    for cls, prob in result['probabilities'].items():
        bar = '█' * int(prob * 20)
        print(f"   {cls:<12} {bar:<20} {prob:.2%}")
