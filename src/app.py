import sys
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from pathlib import Path
import streamlit as st

sys.path.append(str(Path(__file__).parent))
from gradcam import GradCAM

CLASSES    = ['glioma', 'meningioma', 'notumor', 'pituitary']
IMG_SIZE   = 224
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODELS_DIR = Path(__file__).parent.parent / "models"

CLASS_INFO = {
    "glioma": {
        "description": "Gliomas are tumors that occur in the brain and spinal cord. "
                       "They begin in the glial cells that surround nerve cells.",
        "severity": "High"
    },
    "meningioma": {
        "description": "Meningiomas are tumors that arise from the meninges — "
                       "the membranes that surround the brain and spinal cord.",
        "severity": "Moderate"
    },
    "notumor": {
        "description": "No tumor detected. The MRI scan appears normal "
                       "with no signs of abnormal growth.",
        "severity": "None"
    },
    "pituitary": {
        "description": "Pituitary tumors are abnormal growths that develop "
                       "in the pituitary gland at the base of the brain.",
        "severity": "Moderate"
    }
}
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

@st.cache_resource
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
    return model

def predict(image, model):
    img_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs       = model(img_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        class_idx     = probabilities.argmax().item()
        confidence    = probabilities[class_idx].item()

    return CLASSES[class_idx], confidence, probabilities.cpu().numpy()

def get_gradcam(image, model):
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')

    img_tensor = transform(image)
    gradcam    = GradCAM(model)
    heatmap, _ = gradcam.generate(img_tensor)

    heatmap_img = Image.fromarray(np.uint8(255 * heatmap))
    heatmap_img = heatmap_img.resize(image.size, Image.BILINEAR)
    heatmap_np  = np.array(heatmap_img) / 255.0

    colormap = plt.cm.jet(heatmap_np)[:, :, :3]
    img_np   = np.array(image) / 255.0
    overlay  = np.clip(0.6 * img_np + 0.4 * colormap, 0, 1)

    return Image.fromarray(np.uint8(overlay * 255))

def main():
    st.set_page_config(
        page_title="Brain Tumor Detection",
        layout="wide"
    )

    st.title("Brain Tumor Detection from MRI")
    st.markdown("Upload an MRI scan to detect the type of brain tumor "
                "using AI powered by ResNet50.")
    st.divider()

    with st.sidebar:
        st.header("ℹAbout")
        st.markdown("""
        This app uses a deep learning model trained on **7,200 MRI images**
        to classify brain tumors into 4 categories:
        -  **Glioma**
        -  **Meningioma**
        -  **No Tumor**
        -  **Pituitary**
        """)
        st.divider()
        st.markdown("**Model:** ResNet50")
        st.markdown("**Device:** " + str(DEVICE).upper())
        st.markdown("**Developer:** Nadim Hamade")

    try:
        model = load_model()
        st.success("Model loaded successfully!")
    except FileNotFoundError:
        st.error("Model not found! Please train the model first by running train.py")
        st.stop()

    st.subheader("Upload MRI Scan")
    uploaded_file = st.file_uploader(
        "Choose an MRI image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")

        with st.spinner("Analyzing MRI scan..."):
            predicted_class, confidence, probabilities = predict(image, model)
            overlay = get_gradcam(image, model)

        st.divider()
        st.subheader("Results")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.image(image, caption="Original MRI", use_container_width=True)

        with col2:
            st.image(overlay, caption="Grad-CAM Heatmap", use_container_width=True)

        with col3:
            info = CLASS_INFO[predicted_class]
            st.metric("Prediction", predicted_class.capitalize())
            st.metric("Confidence", f"{confidence:.2%}")
            st.markdown(f"**Severity:** {info['severity']}")
            st.markdown(f"**About:** {info['description']}")

        st.divider()
        st.subheader("📈 Class Probabilities")
        for cls, prob in zip(CLASSES, probabilities):
            st.progress(float(prob),
                        text=f"{cls.capitalize():<15} {prob:.2%}")

        st.divider()
        st.warning("This tool is for educational purposes only and should "
                   "not be used as a substitute for professional medical diagnosis.")

if __name__ == "__main__":
    main()
