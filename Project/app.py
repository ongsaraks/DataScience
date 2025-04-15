import asyncio
asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import pandas as pd
import pytesseract  # Install with `pip install pytesseract`
import os
from PIL import Image

# Set page config
st.set_page_config(
    page_title="Product Detection",
    page_icon="🔍",
    layout="centered"
)

# Title and description
st.title("Product Detection and Series Classification")
st.write("Upload an image or take a picture to detect and classify products by their series.")

# Load the model
@st.cache_resource
def load_model():
    model = YOLO(r'C:\Users\gifly\Desktop\DataScience\Project\best.pt')
    return model

# Initialize session state for counts
if 'counts' not in st.session_state:
    st.session_state.counts = {}

# Function to update CSV
def update_csv(series_info):
    csv_path = 'object_counts.csv'

    # Check if the CSV file exists
    if os.path.exists(csv_path):
        # Read the existing CSV file
        df = pd.read_csv(csv_path)
    else:
        # Create a new DataFrame if the file doesn't exist
        df = pd.DataFrame(columns=['Object', 'ID', 'NUM'])

    # Update counts for each detected object and series
    for info in series_info:
        object_name = f"{info['class']} {info['series']}"  # Combine class and series
        object_id = 'D0001' if 'Dryer' in info['class'] else 'I0002'  # Assign ID based on class

        # Check if the object already exists in the CSV
        if object_name in df['Object'].values:
            # Increment the count for the existing object
            df.loc[df['Object'] == object_name, 'NUM'] += 1
        else:
            # Add a new row for the new object
            new_row = {'Object': object_name, 'ID': object_id, 'NUM': 1}
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    # Save the updated DataFrame back to the CSV file
    df.to_csv(csv_path, index=False)

# Function to process image with OCR
def process_image_with_ocr(model, image):
    # Convert to numpy array
    image_np = np.array(image)
    
    # Convert RGB to BGR for OpenCV
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    
    # Perform detection
    results = model(image_bgr)
    
    # Initialize counts and series classification
    counts = {}
    series_info = []  # To store series information for each detected object
    
    # Process detections
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Get class name
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            
            # Extract bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            
            # Crop the detected region for OCR
            cropped_region = image_bgr[y1:y2, x1:x2]
            
            # Perform OCR on the cropped region
            ocr_result = pytesseract.image_to_string(cropped_region, config='--psm 6')
            ocr_result = ocr_result.strip()  # Clean up the OCR result
            
            # Store the series information
            series_info.append({'class': class_name, 'series': ocr_result, 'confidence': conf})
            
            # Increment count for the specific object and series
            object_name = f"{class_name} {ocr_result}"
            counts[object_name] = counts.get(object_name, 0) + 1
            
            # Draw bounding box and label
            cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{class_name} {ocr_result} ({conf:.2f})"
            cv2.putText(image_bgr, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Convert back to RGB for display
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    return image_rgb, counts, series_info

# Create tabs for different input methods
tab1, tab2 = st.tabs(["📷 Take Photo", "📤 Upload Image"])

with tab1:
    st.write("Use your device's camera to take a picture")
    # Camera input
    picture = st.camera_input("Take a picture")
    
    if picture:
        # Convert the image to PIL format
        image = Image.open(picture)
        
        # Load the model
        model = load_model()
        
        if st.button("Detect Objects (Camera)", use_container_width=True):
            with st.spinner("Processing..."):
                # Process the image
                processed_image, counts, series_info = process_image_with_ocr(model, image)
                
                # Update session state
                st.session_state.counts = counts
                
                # Update CSV
                update_csv(series_info)
                
                # Display results
                st.image(processed_image, caption="Detected Objects with Series", use_column_width=True)
                
                # Display counts
                st.subheader("Object Counts")
                st.write(counts)
                
                # Display series information
                st.subheader("Series Information")
                st.write(series_info)

with tab2:
    st.write("Upload an image from your device")
    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Load the model
        model = load_model()
        
        # Read the image
        image = Image.open(uploaded_file)
        
        if st.button("Detect Objects (Upload)", use_container_width=True):
            with st.spinner("Processing..."):
                # Process the image
                processed_image, counts, series_info = process_image_with_ocr(model, image)
                
                # Update session state
                st.session_state.counts = counts
                
                # Update CSV
                update_csv(series_info)
                
                # Display results
                st.image(processed_image, caption="Detected Objects with Series", use_column_width=True)
                
                # Display counts
                st.subheader("Object Counts")
                st.write(counts)
                
                # Display series information
                st.subheader("Series Information")
                st.write(series_info)

# Display current counts in sidebar
st.sidebar.title("Current Counts")
for obj, count in st.session_state.counts.items():
    st.sidebar.metric(obj, count)

# Add download button for CSV
if os.path.exists('object_counts.csv'):
    with open('object_counts.csv', 'rb') as f:
        st.sidebar.download_button(
            label="Download CSV",
            data=f,
            file_name='object_counts.csv',
            mime='text/csv'
        )