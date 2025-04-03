# --- Streamlit Deployment Tutorial: Cats vs. Dogs Image Classification ---

# This tutorial demonstrates how to deploy a pre-trained image classification model (specifically, the
# cats_dogs_model.pth we saved in the previous tutorial) using Streamlit.

import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
import os

# 1. Load the Trained Model

# Load the saved model (make sure the model definition is available in your Streamlit app)
# You need to have the same model architecture definition as in your training script.
# For example, if you trained with EfficientNet-B0:

#model = torch.hub.load('pytorch/vision:v0.10.0', 'efficientnet_b0', pretrained=False)
model = torch.hub.load('pytorch/vision:main', 'efficientnet_b0', pretrained=False)
num_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_features, 2)  # 2 classes (cat, dog)
model.load_state_dict(torch.load('cats_dogs_model.pth', map_location=torch.device('cpu'))) # Load to CPU
model.eval()

# Define the image transformations (same as in training/validation)
transform = T.Compose([
    T.Resize(256),       # Resize for EfficientNet
    T.CenterCrop(224),   # Center crop for consistent input size
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet stats
])

classes = ['cat', 'dog']  # Class names (same as in training)

# 2. Create the Streamlit App

st.title("Cats vs. Dogs Image Classifier")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Classify"):
        with st.spinner("Classifying..."):  # Show a spinner while processing
            input_tensor = transform(image).unsqueeze(0)  # Add batch dimension

            with torch.no_grad():
                output = model(input_tensor)
                probabilities = torch.nn.functional.softmax(output[0], dim=0) # Softmax for probabilities
                predicted_class_index = torch.argmax(probabilities).item()
                predicted_class = classes[predicted_class_index]
                confidence = probabilities[predicted_class_index].item() * 100

            st.header("Prediction")
            st.write(f"The image is a {predicted_class} with {confidence:.2f}% confidence.")

            # Display probabilities for each class (optional)
            st.subheader("Class Probabilities")
            for i, class_name in enumerate(classes):
              st.write(f"{class_name}: {probabilities[i].item()*100:.2f}%")

# 3. Running the Streamlit App

# Save the code as app.py
# Open your terminal, navigate to the directory where you saved app.py
# Run: streamlit run app.py