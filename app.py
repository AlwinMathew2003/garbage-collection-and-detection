import streamlit as st
import torch
import time
from PIL import Image
from ultralytics import YOLO

# Load YOLOv11 Model (Ensure 'best.pt' is in the same directory)
model = YOLO("best.pt")

# Set Streamlit page config
st.set_page_config(
    page_title="YOLOv11 - Object Detection",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .sidebar-title {
        font-size: 28px;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 20px;
    }
    .main-header {
        font-size: 50px;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 30px;
    }
    .sub-header {
        font-size: 35px;
        font-weight: bold;
        color: #2ca02c;
        margin-top: 20px;
        margin-bottom: 10px;
    }
    .content {
        font-size: 28px;
        line-height: 1.6;
        margin-bottom: 20px;
    }
    .footer {
        font-size: 25px;
        text-align: center;
        margin-top: 30px;
        color: #666;
    }
    .highlight {
        background-color: #f0f8ff;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Sidebar with Key Insights ---
st.sidebar.markdown('<div class="sidebar-title">🔍 Quick Info</div>', unsafe_allow_html=True)
st.sidebar.write("**What is YOLO?**")
st.sidebar.write("YOLO (You Only Look Once) is a state-of-the-art real-time object detection system.")
st.sidebar.write("It detects objects in a single forward pass, making it extremely fast compared to traditional methods.")

st.sidebar.write("**Why is YOLO powerful?**")
st.sidebar.write("- Faster than traditional methods 🚀\n"
                 "- Works well for real-time detection 🎥\n"
                 "- Great for self-driving cars, security, & robotics 🤖")

st.sidebar.write("**Model Performance Metrics** (Visuals Below 👇)")
st.sidebar.image("confusion_matrix.png", caption="Confusion Matrix", use_container_width=True)
st.sidebar.image("PR_curve.png", caption="Precision-Recall Curve", use_container_width=True)
st.sidebar.image("F1_curve.png", caption="F1 Score Curve", use_container_width=True)
# --- Main Content ---
st.markdown('<div class="main-header">YOLOv11: Real-Time Object Detection</div>', unsafe_allow_html=True)

# --- Section 1: Image and YOLOv11 Explanation Side by Side ---
st.markdown('<div class="content">', unsafe_allow_html=True)

col1, col2 = st.columns([1, 2])


# YOLOv11 Explanation

st.write("""
**YOLOv11 (You Only Look Once)** is the latest version of the well-known YOLO object detection model.  
It is optimized for speed and accuracy, making it one of the best choices for **real-time object detection in videos and images.**  """)
st.write(  """  ### 🛠️ **How YOLOv11 Works?**
1️⃣ **Splits the image into a grid** (Each cell predicts objects)  
2️⃣ **Detects multiple bounding boxes per grid**  
3️⃣ **Classifies objects & assigns confidence scores**  
4️⃣ **Non-Maximum Suppression (NMS) removes duplicate boxes**  
5️⃣ **Final image is displayed with detected objects**  
""")

# Highlighted section
st.markdown('<div class="highlight">', unsafe_allow_html=True)
st.write("**Did you know?** YOLOv11 can process over **60 frames per second (FPS)** on a standard GPU, making it ideal for real-time applications like video surveillance and autonomous driving.")
st.markdown('</div>', unsafe_allow_html=True)

# --- Section 2: Applications and Architecture ---
st.markdown('<div class="sub-header">🚀 Applications of YOLOv11</div>', unsafe_allow_html=True)
st.write("YOLOv11 is widely used across various industries due to its speed and accuracy. Here are some key applications:")

# Columns for application images
col3, col4, col5 = st.columns(3)

with col3:
    st.image("https://cdn-ilajckf.nitrocdn.com/utLabjbGVjpaYDQkazoKnooguTzYeQRR/assets/images/optimized/rev-481c316/tamediacdn.techaheadcorp.com/wp-content/uploads/2023/10/16044531/19184615_6101008-scaled.jpg", caption="Autonomous Vehicles", use_container_width=True)
    st.write("**Autonomous Vehicles**: YOLOv11 helps self-driving cars detect pedestrians, vehicles, and traffic signs in real-time.")

with col4:
    st.image("https://optexpinnacle.sgp1.cdn.digitaloceanspaces.com/wp-content/uploads/2024/07/29163010/Survillance-System.jpg", caption="Surveillance Systems", use_container_width=True)
    st.write("**Surveillance**: YOLOv11 is used in security systems to detect intruders, track movements, and analyze behavior.")

with col5:
    st.image("https://www.technetexperts.com/wp-content/uploads/2024/08/AI-And-Robotics.jpg", caption="Robotics and AI", use_container_width=True)
    st.write("**Robotics**: Robots use YOLOv11 for object recognition, navigation, and interaction in dynamic environments.")

# --- Section 3: YOLOv11 Architecture ---
st.markdown('<div class="sub-header">🏗️ YOLOv11 Architecture</div>', unsafe_allow_html=True)
st.write("The architecture of YOLOv11 is designed for speed and accuracy. Here's how it works:")

# Display the YOLOv11 architecture image
st.image("https://miro.medium.com/v2/resize:fit:1100/format:webp/1*L8rMuwurmyBH1ixIqcrMSQ.png", caption="YOLOv11 Architecture", use_container_width=True)

# --- Section 4: Try Object Detection ---
st.markdown('<div class="sub-header">📸 Try Object Detection!</div>', unsafe_allow_html=True)
st.write("Upload an image and see YOLOv11 in action. It will detect objects in real-time and display the results.")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    col6, col7 = st.columns(2)

    with col6:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("🔍 Detect Objects", key="detect", use_container_width=True):
        with col7:
            st.write("Processing image...")
            time.sleep(2)
            results = model.predict(image, save=False, imgsz=640)
            detected_img = results[0].plot()  # Get the detected image
            st.image(detected_img, caption="Detected Objects", use_container_width=True)

# --- Footer ---
st.markdown("---")
st.markdown('<div class="footer">💡 **Powered by YOLOv11 | Built with Streamlit**</div>', unsafe_allow_html=True)