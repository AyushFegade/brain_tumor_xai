import numpy as np
import streamlit as st
import torch
from PIL import Image
import matplotlib.pyplot as plt
from src.core import generate_lime_explanation
from src.core import (
    GradCAM, get_gradcam_target_layer, load_bundle,
    preprocess_pil, softmax_with_temperature
)

st.set_page_config(page_title="Brain Tumor MRI Classifier", layout="wide")
st.title("Brain Tumor MRI Classification — Explainable & Calibrated")
st.caption("Research prototype. Not for clinical diagnosis.")

bundle_path = st.sidebar.text_input("Model bundle", "outputs/best_model_bundle.pth")

@st.cache_resource
def load_model(path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, meta = load_bundle(path, device)
    return model, meta, device

try:
    model, meta, device = load_model(bundle_path)
except Exception as e:
    st.warning(f"Train/export the model first. Could not load bundle: {e}")
    st.stop()

uploaded = st.file_uploader("Upload a brain MRI image", type=["jpg", "jpeg", "png"])
if uploaded:
    image = Image.open(uploaded).convert("RGB")
    x = preprocess_pil(image, meta["image_size"]).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = softmax_with_temperature(logits.cpu(), meta["temperature"])[0].numpy()

    pred_idx = int(np.argmax(probs))
    confidence = float(np.max(probs))
    pred_name = meta["class_names"][pred_idx]
    uncertain = confidence < meta["confidence_threshold"]

    c1, c2 = st.columns(2)
    with c1:
        st.image(image, caption="Uploaded MRI", use_container_width=True)
    with c2:
        st.subheader("Prediction")
        st.metric("Class", pred_name)
        st.metric("Calibrated confidence", f"{confidence:.1%}")
        if uncertain:
            st.error("UNCERTAIN — expert review recommended")
        else:
            st.success("Accepted by the selective-prediction threshold")

        st.write("Class probabilities")
        st.dataframe({
            "class": meta["class_names"],
            "probability": [float(p) for p in probs]
        }, use_container_width=True)

    # Grad-CAM
    target = get_gradcam_target_layer(model, meta["architecture"])
    cam_engine = GradCAM(model, target)
    cam, _ = cam_engine(x, pred_idx)
    cam_engine.close()

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image.resize((meta["image_size"], meta["image_size"])))
    ax.imshow(cam, alpha=0.4)
    ax.axis("off")
    ax.set_title(f"Grad-CAM — {pred_name}")
    st.pyplot(fig)

    # LIME Explanation integration
    st.divider()
    st.subheader("LIME Explanation")
    st.write("LIME perturbs the image to identify which specific regions contributed most to the model's decision.")

    with st.spinner("Generating LIME Explanation (this takes a few seconds)..."):
        # We pass x (the preprocessed image tensor) directly 
        lime_result = generate_lime_explanation(model, x, pred_idx, device)
        st.image(lime_result, caption="LIME Mask", use_container_width=True, clamp=True)

st.divider()
st.caption(
    "The confidence value is temperature-calibrated on validation data. "
    "The uncertainty threshold is learned from validation confidence to target a chosen coverage, "
    "rather than being an arbitrary fixed number."
)