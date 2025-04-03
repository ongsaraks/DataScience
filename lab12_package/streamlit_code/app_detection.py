# --- Streamlit Deployment Tutorial: YOLOv8 Object Detection (Animals) ---

# This tutorial demonstrates how to deploy a pre-trained YOLOv8 object detection model on Streamlit,
# focusing on detecting a specific set of object classes (animals). We download the model directly
# from the internet.

import streamlit as st
import torch
from ultralytics import YOLO
from PIL import Image, ImageFont, ImageDraw
import numpy as np

# 1. Load the Pre-trained YOLOv8 Model

# Download and load the pre-trained YOLOv8n model (or any other size you prefer).
# YOLO will automatically download it if it's not already cached.
model = YOLO('yolov8n.pt')  # You can also use yolov8s.pt, yolov8m.pt, etc.

# Define the animal classes of interest (adjust as needed)
animal_classes = ['cat', 'dog', 'bird', 'horse', 'cow', 'sheep', 'elephant', 'bear', 'zebra', 'giraffe']

# Get the class names from the model
class_names = model.names
print("Class names:"+str(class_names))

# Create a mapping from animal class names to their IDs in YOLO
#animal_class_ids = [class_names.index(animal) for animal in animal_classes if animal in class_names]
animal_class_ids = []
for animal in animal_classes:
    for k,v in class_names.items():
        if v == animal:
            animal_class_ids.append(k)
            break
#animal_class_ids.append([key for key, val in class_names.items() if val == animal])
#for animal in animal_classes:
#    if animal in class_names:
#        animal_class_ids.append(class_names.index(animal))
print("Animal IDS"+str(animal_class_ids))

# 2. Create the Streamlit App

st.title("Animal Detection with YOLOv8")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Detect Animals"):
        with st.spinner("Detecting animals..."):
            results = model(image)

            # Process the results and filter for animal classes
            detected_animals = []
            for result in results:
                boxes = result.boxes  # Boxes object for detected objects
                for box in boxes:
                    print("HEHE"+str(box))
                    cls = int(box.cls[0])  # Class ID
                    print("CLS = "+str(cls))
                    if cls in animal_class_ids: # Check if it's an animal
                        xyxy = box.xyxy[0].tolist() # Bounding box coordinates (x1, y1, x2, y2)
                        conf = float(box.conf[0])  # Confidence score
                        label = class_names[cls] # Class name
                        detected_animals.append({"label": label, "bbox": xyxy, "conf": conf})

            # Display the results
            if detected_animals:
                st.header("Detected Animals")
                for animal in detected_animals:
                    st.write(f"{animal['label']} (Confidence: {animal['conf']:.2f})")

                # (Optional) Draw bounding boxes on the image
                annotated_image = image.copy() # Create a copy to avoid modifying the original
                #draw = Image.Draw(annotated_image)
                draw = ImageDraw.Draw(annotated_image)
                for animal in detected_animals:
                    x1, y1, x2, y2 = animal["bbox"]
                    draw.rectangle([(x1, y1), (x2, y2)], outline="red", width=2) # Draw bounding box
                    draw.text((x1, y1 - 10), f"{animal['label']} {animal['conf']:.2f}", fill="red") # Add label

                st.image(annotated_image, caption="Image with Bounding Boxes", use_column_width=True)

            else:
                st.write("No animals detected in the image.")



# 3. Running the Streamlit App

# Save the code as app.py
# Open your terminal, navigate to the directory where you saved app.py
# Run: streamlit run app.py