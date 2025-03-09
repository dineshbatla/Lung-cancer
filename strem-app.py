import streamlit as st
import tensorflow as tf
import gdown
import numpy as np
import cv2
from PIL import Image
import io

1y3M7qBImeN_IbxjYEpNRMc2HBWkuE9eQ
file_id = "1y3M7qBImeN_IbxjYEpNRMc2HBWkuE9eQ"
# Path to save the model
model_path = "model.h5"

# Function to download and load the model
@st.cache_resource
def load_model():
    # Download model from Google Drive
    gdown.download(f"https://drive.google.com/uc?id={file_id}", model_path, quiet=False)
    
    # Load the model
    model = tf.keras.models.load_model(model_path)
    return model

st.title("Lung Cancer Detection")

# Load the model
model = load_model()

st.write("✅ Model Loaded Successfully!")

# Image Preprocessing Function
def preprocess_image(uploaded_file):
    # Convert uploaded file to a numpy array
    image = Image.open(uploaded_file).convert("L")  # Convert to grayscale
    image = image.resize((128, 128))  # Resize to match model input size
    img = np.array(image)  # Convert to NumPy array
    img = np.expand_dims(img, axis=-1)  # Add a channel dimension to make it (128, 128, 1)
    img = np.expand_dims(img, axis=0)  # Add batch dimension to make it (1, 128, 128, 1)
    img = img / 255.0  # Normalize pixel values
    return img

# Streamlit UI
st.title("Lung Cancer Classification")
st.write("Upload a CT scan image to classify it as **Benign, Malignant, or Normal**.")

uploaded_file = st.file_uploader("Choose a CT scan image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if model:
        # Preprocess and predict
        processed_image = preprocess_image(uploaded_file)
        predictions = model.predict(processed_image)
        
        class_names = ["Benign", "Malignant", "Normal"]
        predicted_class = class_names[np.argmax(predictions)]
        confidence = np.max(predictions)

        # Show results
        st.subheader(f"Prediction: {predicted_class}")
        st.write(f"Confidence: {confidence:.2f}")

