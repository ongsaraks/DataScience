import asyncio
asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import pandas as pd
import tempfile
import os
from PIL import Image
import io

# Set page config
st.set_page_config(
    page_title="Product Detection",
    page_icon="🔍",
    layout="wide"  # Center the layout for better mobile experience
)

# Title and description
st.title("Product Detection and Counting")
st.write("Upload an image or take a picture to detect and count WaterPump and AirPump")

# Load the model
@st.cache_resource
def load_model():
    model = YOLO(r'C:\Users\gifly\Desktop\DataScience\Project\best.pt')
    return model

# Initialize session state for counts
if 'counts' not in st.session_state:
    st.session_state.counts = {'SOBO_WP_4200': 0, 'YAMANO_LPP_20': 0}

# Function to update CSV
def update_csv(new_counts):
    # Check if the CSV file exists
    if os.path.exists('object_counts.csv'):
        # Read the existing CSV file
        df = pd.read_csv('object_counts.csv')
        # Update the counts by adding the new counts
        df.loc[df['Object'] == 'SOBO_WP_4200', 'NUM'] += new_counts['SOBO_WP_4200']
        df.loc[df['Object'] == 'YAMANO_LPP_20', 'NUM'] += new_counts['YAMANO_LPP_20']
    else:
        # Create a new DataFrame if the file doesn't exist
        df = pd.DataFrame({
            'Object': ['SOBO_WP_4200', 'YAMANO_LPP_20'],
            'ID': ['WP0010', 'AP0020'],
            'NUM': [new_counts['SOBO_WP_4200'], new_counts['YAMANO_LPP_20']]
        })
    
    # Save the updated DataFrame back to the CSV file
    df.to_csv('object_counts.csv', index=False)

# Function to process image
def process_image(model, image):
    # Convert to numpy array
    image_np = np.array(image)
    
    # Convert RGB to BGR for OpenCV
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    
    # Perform detection
    results = model(image_bgr)
    
    # Initialize counts
    counts = {'SOBO_WP_4200': 0, 'YAMANO_LPP_20': 0}
    
    # Process detections
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Get class name
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            
            # Increment count
            counts[class_name] += 1
            
            # Extract bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            
            # Draw a thicker rectangle (bounding box)
            cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (0, 255, 0), thickness=5)
            
            # Add label directly on the bounding box
            label = f"{class_name} {conf:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(image_bgr, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), (0, 255, 0), -1)
            cv2.putText(image_bgr, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    # Add total counts to image
    count_text = f"SOBO_WP_4200: {counts['SOBO_WP_4200']}, YAMANO_LPP_20: {counts['YAMANO_LPP_20']}"
    cv2.putText(image_bgr, count_text, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Convert back to RGB for display
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    return image_rgb, counts

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
        
        if st.button("Detect Objects (Camera)", use_container_width=True):  # Button spans full width
            with st.spinner("Processing..."):
                # Process the image
                processed_image, counts = process_image(model, image)
                
                # Update session state
                st.session_state.counts = counts
                
                # Update CSV
                update_csv(counts)
                
                # Display results
                st.image(processed_image, caption="Detected Objects", use_container_width=True)
                
                # Display counts
                st.subheader("Object Counts")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("SOBO_WP_4200", counts['SOBO_WP_4200'])
                with col2:
                    st.metric("YAMANO_LPP_20", counts['YAMANO_LPP_20'])
                
                # Display CSV content
                st.subheader("CSV Data")
                df = pd.read_csv('object_counts.csv')
                st.dataframe(df)

with tab2:
    st.write("Upload an image from your device")
    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Load the model
        model = load_model()
        
        # Read the image
        image = Image.open(uploaded_file)
        
        if st.button("Detect Objects (Upload)", use_container_width=True):  # Button spans full width
            with st.spinner("Processing..."):
                # Process the image
                processed_image, counts = process_image(model, image)
                
                # Update session state
                st.session_state.counts = counts
                
                # Update CSV
                update_csv(counts)
                
                # Display results
                st.image(processed_image, caption="Detected Objects", use_container_width=True)
                
                # Display counts
                st.subheader("Object Counts")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("SOBO_WP_4200", counts['SOBO_WP_4200'])
                with col2:
                    st.metric("YAMANO_LPP_20", counts['YAMANO_LPP_20'])
                
                # Display CSV content
                st.subheader("CSV Data")
                df = pd.read_csv('object_counts.csv')
                st.dataframe(df)

# Display current counts in sidebar
st.sidebar.title("Current Counts")
st.sidebar.metric("SOBO_WP_4200", st.session_state.counts['SOBO_WP_4200'])
st.sidebar.metric("YAMANO_LPP_20", st.session_state.counts['YAMANO_LPP_20'])

# Add download button for CSV
if os.path.exists('object_counts.csv'):
    with open('object_counts.csv', 'rb') as f:
        st.sidebar.download_button(
            label="Download CSV",
            data=f,
            file_name='object_counts.csv',
            mime='text/csv'
        )

# Add manual adjustment buttons in the sidebar
st.sidebar.title("Adjust Counts Manually")

# Function to adjust counts
def adjust_count(object_name, adjustment):

    # Check if the CSV file exists

    df = pd.read_csv('object_counts.csv')

        # Update the count for the specific object
    if object_name in df['Object'].values:
        df.loc[df['Object'] == object_name, 'NUM'] += adjustment


    # Save the updated DataFrame back to the CSV file
    df.to_csv('object_counts.csv', index=False)

# Manual adjustment for SOBO_WP_4200
st.sidebar.subheader("SOBO_WP_4200")
col1, col2 = st.sidebar.columns(2)
with col1:
    if st.sidebar.button("➕ Increase SOBO_WP_4200", key="increase_SOBO"):
        adjust_count("SOBO_WP_4200", 1)
with col2:
    if st.sidebar.button("➖ Decrease SOBO_WP_4200", key="decrease_SOBO"):
        adjust_count("SOBO_WP_4200", -1)

# Manual adjustment for YAMANO_LPP_20
st.sidebar.subheader("YAMANO_LPP_20")
col3, col4 = st.sidebar.columns(2)
with col3:
    if st.sidebar.button("➕ Increase YAMANO_LPP_20", key="increase_YAMANO"):
        adjust_count("YAMANO_LPP_20", 1)
with col4:
    if st.sidebar.button("➖ Decrease YAMANO_LPP_20", key="decrease_YAMANO"):
        adjust_count("YAMANO_LPP_20", -1)