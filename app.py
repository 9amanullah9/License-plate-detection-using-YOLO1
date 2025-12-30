import streamlit as st
import ultralytics
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2

# Page Config
st.set_page_config(page_title="License Plate Recognition", page_icon="🚗")

st.title("🚗 License Plate Detection")
st.write("Upload a vehicle image to detect and crop license plates.")

# --- Sidebar for Model Selection ---
st.sidebar.header("Model Settings")
model_source = st.sidebar.radio("Select Model Source", ["Use local 'best.pt'", "Upload .pt file"])
model = None

# Logic to load the model
try:
    if model_source == "Use local 'best.pt'":
        try:
            # Ensure you have the 'best.pt' from your LICENSE PLATE training in the folder
            model = YOLO('best.pt') 
            st.sidebar.success("Loaded local 'best.pt'!")
        except Exception:
            st.sidebar.error("Could not find 'best.pt'. Please upload it.")
            
    elif model_source == "Upload .pt file":
        model_file = st.sidebar.file_uploader("Upload your trained .pt file", type=['pt'])
        if model_file is not None:
            with open("temp_lp_model.pt", "wb") as f:
                f.write(model_file.getbuffer())
            model = YOLO("temp_lp_model.pt")
            st.sidebar.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")

# --- Main Image Upload ---
uploaded_file = st.file_uploader("Choose a vehicle image...", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None and model is not None:
    # 1. Display Original
    image = Image.open(uploaded_file)
    st.image(image, caption="Original Image", use_container_width=True)
    
    # 2. Inference
    if st.button("Detect License Plate"):
        with st.spinner('Scanning for plates...'):
            try:
                # Run prediction
                results = model.predict(image, conf=0.25)
                
                # Plot bounding boxes on the main image
                res_plotted = results[0].plot()
                st.subheader("Detection Result")
                st.image(res_plotted[:, :, ::-1], caption="Detected Plates", use_container_width=True)
                
                # 3. Crop and Show Individual Plates
                boxes = results[0].boxes
                if len(boxes) > 0:
                    st.write("### 🔍 Extracted Plates")
                    
                    # Convert original PIL image to numpy array for cropping
                    img_array = np.array(image)
                    
                    # Loop through detections
                    for i, box in enumerate(boxes):
                        # Get coordinates (x1, y1, x2, y2)
                        coords = box.xyxy[0].cpu().numpy().astype(int)
                        x1, y1, x2, y2 = coords
                        
                        # Crop the plate from the array
                        cropped_plate = img_array[y1:y2, x1:x2]
                        
                        # Display crop
                        st.image(cropped_plate, caption=f"Plate #{i+1} (Conf: {float(box.conf):.2f})", width=200)
                        
                    st.success(f"Found {len(boxes)} license plate(s).")
                else:
                    st.warning("No license plates detected.")
                    
            except Exception as e:
                st.error(f"Error processing image: {e}")

elif uploaded_file is not None and model is None:
    st.warning("Please upload or load your trained model first!")