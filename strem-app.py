import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import io

# Load the model (ensure the file path is correct)
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model(r"C:\Users\tapan\Downloads\lung_cancer_model.h5")  # Change path if needed
        st.success("Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

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

