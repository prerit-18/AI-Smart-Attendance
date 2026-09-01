import streamlit as st
import os
import sys

# Configure Streamlit page layout at the absolute start
st.set_page_config(
    page_title="AI Smart Attendance",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add base directory to path so imports work correctly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import traceback

try:
    from src.database import init_db
    import src.pages.dashboard as dashboard
    import src.pages.register_student as register_student
    import src.pages.student_database as student_database
    import src.pages.live_attendance as live_attendance
    import src.pages.attendance_records as attendance_records
    import src.pages.analytics as analytics
    import src.pages.model_evaluation as model_evaluation
except Exception as e:
    st.error("### ❌ App Initialization Error")
    st.markdown("An error occurred while importing application components. The full traceback is shown below:")
    st.code(traceback.format_exc())
    
    # Run diagnostics on the imported cv2 module
    try:
        import cv2
        st.markdown("---")
        st.subheader("🔍 OpenCV (cv2) Debug Diagnostics")
        st.write(f"**Loaded from path:** `{getattr(cv2, '__file__', 'None')}`")
        st.write(f"**Package folder (`__path__`):** `{getattr(cv2, '__path__', 'None')}`")
        st.write(f"**Attributes:**")
        st.code(str(dir(cv2)))
    except Exception as cv2_err:
        st.error(f"Failed to import cv2: {cv2_err}")
    st.stop()


# 1. Initialize Database on app start
init_db()

# 2. Sidebar Layout
st.sidebar.title("🎓 AI Smart Attendance")
st.sidebar.markdown("*Deep Learning Powered Attendance*")
st.sidebar.markdown("---")

# Demo Mode Toggle
demo_mode = st.sidebar.checkbox(
    "🔬 Enable Demo Mode", 
    value=False,
    help="Toggles simulated webcam/data inputs so you can explore the application without physical hardware."
)

# Model Selection Dropdown
selected_model = st.sidebar.selectbox(
    "🧠 Active Face Model",
    ["mobilenet", "custom_cnn", "custom_cnn_baseline"],
    index=0,
    format_func=lambda x: "MobileNetV2 (Transfer Learning)" if x == "mobilenet" else (
        "Improved CNN (Trained)" if x == "custom_cnn" else "Baseline CNN (Trained)"
    ),
    help="Select the Deep Learning model to use. Face embeddings are model-specific; you must use the same model for both registration and live recognition."
)

# Synchronize biometric embeddings for active model if needed (safe, non-destructive)
try:
    from src.embeddings import sync_student_embeddings
    sync_student_embeddings(selected_model)
    from src.face_recognition import load_known_face_embeddings
    load_known_face_embeddings(model_type=selected_model, force_reload=True)
except Exception as sync_err:
    pass

st.sidebar.markdown("---")
# Navigation Menu
nav_selection = st.sidebar.radio(
    "Navigation Menu",
    [
        "📊 Dashboard",
        "👤 Register Student",
        "👥 Student Database",
        "🎥 Live Attendance",
        "📋 Attendance Records",
        "📈 Analytics & Sequence",
        "🎯 Model Evaluation",
        "ℹ️ About Project"
    ]
)

# 3. Route Nav Selection to Pages
if nav_selection == "📊 Dashboard":
    dashboard.show()
elif nav_selection == "👤 Register Student":
    register_student.show(demo_mode=demo_mode, model_type=selected_model)
elif nav_selection == "👥 Student Database":
    student_database.show(demo_mode=demo_mode, model_type=selected_model)
elif nav_selection == "🎥 Live Attendance":
    live_attendance.show(demo_mode=demo_mode, model_type=selected_model)
elif nav_selection == "📋 Attendance Records":
    attendance_records.show()
elif nav_selection == "📈 Analytics & Sequence":
    analytics.show()
elif nav_selection == "🎯 Model Evaluation":
    model_evaluation.show()
elif nav_selection == "ℹ️ About Project":
    st.title("ℹ️ About AI Smart Attendance")
    st.markdown(
        """
        ### Project Overview
        The **AI Smart Attendance System** is an end-to-end computer-vision and sequence analytics project 
        designed as a capstone application for the **Deep Learning Specialization**. 
        It integrates convolutional neural networks for face recognition with recurrent sequence networks for attendance behavior forecasting.
        
        ---
        
        ### Deep Learning Course Mapping
        
        | Course Module | Deep Learning Concept | System Implementation |
        |---|---|---|
        | **Course 1: Neural Networks** | Deep neural network structure (weights, activation) | Foundation layer for feature classification mapping |
        | **Course 2: Optimization** | Dropout, Batch Normalization, Adam | Implemented in `custom_cnn` to optimize and stabilize training |
        | **Course 3: ML Projects** | Train/Dev/Test splits, Error Analysis | Evaluates models using test partitions and profiles camera errors |
        | **Course 4: CNNs** | Feature representations & Cosine similarity | Pretrained MobileNetV2 base extracts 128D face feature vectors |
        | **Course 5: Sequence Models** | LSTMs / GRUs recurrent networks | Constructs temporal vectors of history to predict future attendance |
        
        ---
        
        ### Responsible Design & Biometric Privacy Safeguards
        Because this prototype processes facial data, it strictly respects fundamental privacy principles:
        1. **Local Storage Execution**: All images, SQLite database tables, and model weight binaries are stored entirely locally on the student's laptop.
        2. **Minimalist Data Capture**: The system captures only cropped facial frames and stores them strictly inside a secure gitignored directory (`data/students/`). No personally identifiable information is transmitted off the system.
        3. **GDPR 'Right to Erasure' compliance**: Deleting a student via the records page runs a cascaded SQLite delete query and recursively deletes all raw sample files on disk.
        4. **Academic Warning**: This application is a prototype demonstrating machine learning concepts. It must not be deployed in production environments without performing a Biometric Impact Assessment and obtaining formal consent.
        
        ---
        
        ### Core Technology Stack
        - **Frontend**: Streamlit 1.62.0 (python UI dashboard)
        - **Deep Learning**: TensorFlow/Keras 2.21.0
        - **Computer Vision**: OpenCV-Python 5.0.0
        - **Data Handling**: Pandas & NumPy
        - **Database**: SQLite3
        """
    )
