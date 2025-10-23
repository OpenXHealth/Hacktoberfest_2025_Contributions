# app.py
# Streamlit app to upload an X-ray, show prediction and Grad-CAM heatmap.

import streamlit as st
from predict import load_trained_model, predict_image, preprocess_image
from gradcam import get_img_array, make_gradcam_heatmap, overlay_heatmap
import numpy as np
import cv2
import os
from PIL import Image
import tempfile

st.set_page_config(page_title="LungScanAI", layout="centered")

st.title("🫁 LungScanAI — Pneumonia Detector")
st.markdown("Upload a chest X-ray image and the model will predict **Pneumonia** vs **Normal** and show a Grad-CAM heatmap.")

MODEL_PATH = os.path.join("model", "lung_model.h5")

@st.cache_resource
def load_model_cached():
    try:
        return load_trained_model(MODEL_PATH)
    except Exception as e:
        st.error(f"Model load error: {e}")
        return None

model = load_model_cached()

upload = st.file_uploader("Upload chest X-ray (jpg/png)", type=["jpg", "jpeg", "png"])

if upload:
    file_bytes = np.asarray(bytearray(upload.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    st.image(rgb, caption="Uploaded Image", use_column_width=True)

    if model is None:
        st.warning("Model not available. Place trained model at model/lung_model.h5 or run train_model.py.")
    else:
        # Save temp file
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        Image.fromarray(rgb).save(tfile.name)

        result = predict_image(model, tfile.name)
        st.markdown(f"**Prediction:** {result['class']}  \n**Confidence:** {result['confidence']:.3f}")

        # Grad-CAM
        try:
            img_array = get_img_array(tfile.name, size=(224,224))
            heatmap = make_gradcam_heatmap(img_array, model)
            overlayed = overlay_heatmap(tfile.name, heatmap)
            overlayed_rgb = cv2.cvtColor(overlayed, cv2.COLOR_BGR2RGB)
            st.image(overlayed_rgb, caption="Grad-CAM Heatmap", use_column_width=True)
        except Exception as e:
            st.error(f"Grad-CAM error: {e}")

        os.unlink(tfile.name)
else:
    st.info("Upload an X-ray to get started. If you don't have a trained model, run `train_model.py` or place a model at model/lung_model.h5.")
