import os
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
from pathlib import Path
 
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
 
 
class GradCAM:
    def __init__(self, model):
        self.model       = model
        self.gradients   = None
        self.activations = None
 
        target_layer = model.layer4[-1]
        target_layer.register_forward_hook(self._save_activations)
        target_layer.register_full_backward_hook(self._save_gradients)
 
    def _save_activations(self, module, input, output):
        self.activations = output.detach()
 
    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
 
    def generate(self, image_tensor, class_idx=None):
        self.model.eval()
        image_tensor = image_tensor.unsqueeze(0).to(DEVICE)
 
        output = self.model(image_tensor)
 
        if class_idx is None:
            class_idx = output.argmax(1).item()
 
        self.model.zero_grad()
        output[0, class_idx].backward()
 
        weights  = self.gradients.mean(dim=[2, 3], keepdim=True)
        heatmap  = (weights * self.activations).sum(dim=1).squeeze()
        heatmap  = F.relu(heatmap)
        heatmap  = heatmap / (heatmap.max() + 1e-8)
 
        return heatmap.cpu().numpy(), class_idx
 
 
def visualize_gradcam(image_path, model, save_path=None):
    img        = Image.open(image_path).convert("RGB")
    img_tensor = transform(img)
 
    gradcam    = GradCAM(model)
    heatmap, class_idx = gradcam.generate(img_tensor)
 
    heatmap_img = Image.fromarray(np.uint8(255 * heatmap))
    heatmap_img = heatmap_img.resize(img.size, Image.BILINEAR)
    heatmap_np  = np.array(heatmap_img) / 255.0
 
    colormap = plt.cm.jet(heatmap_np)[:, :, :3]
    img_np   = np.array(img) / 255.0
    overlay  = 0.6 * img_np + 0.4 * colormap
 
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Grad-CAM — Predicted: {CLASSES[class_idx].capitalize()}",
                 fontsize=16, fontweight='bold')
 
    axes[0].imshow(img)
    axes[0].set_title("Original MRI")
    axes[0].axis('off')
 
    axes[1].imshow(heatmap_np, cmap='jet')
    axes[1].set_title("Heatmap")
    axes[1].axis('off')
 
    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")
    axes[2].axis('off')
 
    plt.tight_layout()
 
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Grad-CAM saved → {save_path}")
 
    plt.show()
    return CLASSES[class_idx]
 
 
if __name__ == "__main__":
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
    print("ResNet50 model loaded!")
 
    cls        = random.choice(CLASSES)
    sample_dir = Path(f"../data/testing/{cls}")
    test_image = sample_dir / random.choice(os.listdir(sample_dir))
    print(f"Testing on: {cls} — {test_image.name}")
 
    RESULTS_DIR.mkdir(exist_ok=True)
    visualize_gradcam(test_image, model,
                      save_path=RESULTS_DIR / "gradcam_sample.png")
