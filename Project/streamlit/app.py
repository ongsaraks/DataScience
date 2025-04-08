# --- Streamlit Deployment Tutorial: YOLOv8 Object Detection ---

import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# 1. Load the Trained Model

# Path to the best.pt file
model_path = r"C:\Users\gifly\Desktop\DataScience\Project\best.pt"  # Update with your model path

@st.cache_resource
def load_model(path):
    try:
        model = YOLO(path)  # Load the YOLOv8 model
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model(model_path)

# 2. Define the Streamlit App

st.title("YOLOv8 Object Detection")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Run Detection"):
        with st.spinner("Running detection..."):  # Show a spinner while processing
            # Convert the image to a NumPy array
            image_np = np.array(image)

            # Run inference
            results = model.predict(image_np)

            # Display results
            st.header("Detection Results")
            st.image(results[0].plot(), caption="Detected Objects", use_column_width=True)

# 3. Running the Streamlit App

# Save the code as app.py
# Open your terminal, navigate to the directory where you saved app.py
# Run: streamlit run app.py