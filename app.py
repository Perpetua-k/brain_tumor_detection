import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image
import matplotlib.pyplot as plt
import cv2

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Brain Tumor Detection",
    page_icon="🧠",
    layout="centered"
)

# ==========================================
# LOAD MODEL
# ==========================================

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("brain_tumor_model.keras")
    return model

model = load_model()

# ==========================================
# TITLE
# ==========================================

st.title("🧠 Brain Tumor Detection System")

st.write(
    "Upload an MRI image and the AI model will predict whether a brain tumor exists."
)

# ==========================================
# FILE UPLOAD
# ==========================================

uploaded_file = st.file_uploader(
    "Upload MRI Image",
    type=["jpg", "jpeg", "png"]
)

# ==========================================
# PREDICTION
# ==========================================

if uploaded_file is not None:

    # Open image
    img = Image.open(uploaded_file)

    # Display image
    st.image(img, caption="Uploaded MRI Scan", use_container_width=True)

    # Resize image
    img_resized = img.resize((224,224))

    # Convert image to array
    img_array = image.img_to_array(img_resized)

    # Normalize
    img_array = img_array / 255.0

    # Expand dimensions
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)[0][0]

    st.subheader("Prediction Result")

    if prediction > 0.5:
        st.error("Tumor Detected")
    else:
        st.success("No Tumor Detected")

    # ==========================================
    # SIMPLE XAI VISUALIZATION
    # ==========================================

    st.subheader("Explainable AI Visualization")

    # Convert image for visualization
    img_cv = np.array(img_resized)

    gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)

    heatmap = cv2.applyColorMap(gray, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(img_cv, 0.6, heatmap, 0.4, 0)

    fig, ax = plt.subplots(1,2, figsize=(10,5))

    ax[0].imshow(img_cv)
    ax[0].set_title("Original MRI")
    ax[0].axis("off")

    ax[1].imshow(overlay)
    ax[1].set_title("XAI Overlay Visualization")
    ax[1].axis("off")

    st.pyplot(fig)

# ==========================================
# FOOTER
# ==========================================

st.markdown("---")

st.write("Developed using TensorFlow, MobileNetV2, Streamlit, and Explainable AI")
