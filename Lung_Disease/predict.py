# predict.py
# Helpers to load the trained model and predict on a single image.

import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import os

MODEL_PATH = os.path.join("model", "lung_model.h5")
IMG_SIZE = (224, 224)
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]  # adjust if your generator orders differently

def load_trained_model(path=MODEL_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found at {path}. Run training or place a model file there.")
    model = load_model(path)
    return model

def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB").resize(IMG_SIZE)
    arr = np.array(img) / 255.0
    return np.expand_dims(arr, axis=0)

def predict_image(model, image_path):
    x = preprocess_image(image_path)
    probs = model.predict(x)[0]
    idx = int(np.argmax(probs))
    return {"class": CLASS_NAMES[idx], "confidence": float(probs[idx]), "probs": probs.tolist()}
