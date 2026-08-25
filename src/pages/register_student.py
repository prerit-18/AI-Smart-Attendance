import streamlit as st
import cv2
import numpy as np
import os
import sys
from PIL import Image
import time

# Add base directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import STUDENTS_DIR, IMAGE_SIZE
from src.database import register_student, save_embedding, get_student
from src.face_detection import detect_faces, crop_face
from src.preprocessing import check_face_quality, resize_image, normalize_image
from src.embeddings import get_face_model, get_embedding

def show(demo_mode=False, model_type="custom_cnn"):
    st.title("👤 Register Student")
    st.markdown("### Register a new student and enroll their biometrics")
    
    # 1. Registration form fields
    col1, col2 = st.columns(2)
    with col1:
        student_id = st.text_input("Student ID (e.g. ST001)", placeholder="ST001").strip()
        name = st.text_input("Full Name", placeholder="John Doe").strip()
        roll_number = st.text_input("Roll Number", placeholder="2210991100").strip()
    with col2:
        section = st.text_input("Class / Section", placeholder="CSE-A").strip()
        department = st.text_input("Department", placeholder="Computer Science").strip()
        email = st.text_input("Email Address", placeholder="johndoe@university.edu").strip()
        
    st.markdown("---")
    st.subheader("📷 Face Capture & Enrollment")
    
    # Validation flags
    fields_valid = student_id and name and roll_number and section and department and email
    
    if not fields_valid:
        st.info("⚠️ Please fill in all the student details above before capturing faces.")
        return
        
    # Check if student ID already exists
    existing_student = get_student(student_id)
    if existing_student:
        st.error(f"❌ Student ID '{student_id}' is already registered to **{existing_student['name']}**.")
        return
        
    # Create target directory for student's raw images
    student_dir = os.path.join(STUDENTS_DIR, student_id)
    
    # Create navigation tabs for Capture Method
    tab1, tab2 = st.tabs(["📸 Browser Camera (Web App standard)", "🖥️ OpenCV Local Webcam (Direct)"])
    
    # Tab 1: Browser camera input
    with tab1:
        st.write("Capture face images using your browser's webcam.")
        camera_img = st.camera_input("Position your face in the center of the frame")
        
        if camera_img is not None:
            # Check if directory exists, if not create it
            os.makedirs(student_dir, exist_ok=True)
            
            # Read image using PIL
            pil_img = Image.open(camera_img)
            img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            
            # Face Detection
            faces = detect_faces(img_bgr)
            
            if len(faces) == 0:
                st.error("❌ No face detected. Please ensure your face is well-lit and fully visible.")
            elif len(faces) > 1:
                st.error(f"❌ Multiple faces ({len(faces)}) detected. Only one person should be in the frame.")
            else:
                x, y, w, h = faces[0]
                
                # Quality Check
                is_valid, reason = check_face_quality(img_bgr, (x, y, w, h))
                
                # Draw bounding box for visual feedback
                feedback_img = img_bgr.copy()
                box_color = (0, 255, 0) if is_valid else (0, 165, 255) # Green if good, Orange if blurry/small
                cv2.rectangle(feedback_img, (x, y), (x+w, y+h), box_color, 2)
                feedback_rgb = cv2.cvtColor(feedback_img, cv2.COLOR_BGR2RGB)
                
                st.image(feedback_rgb, caption="Face Detection Feedback", width=300)
                
                if not is_valid:
                    st.warning(f"⚠️ Quality check failed: {reason}")
                else:
                    st.success("✅ Face quality is good!")
                    
                    # Capture confirmation button
                    if st.button("Save Sample and Generate Embedding"):
                        # Count existing files to name the sample
                        existing_samples = len([f for f in os.listdir(student_dir) if f.endswith('.png')])
                        sample_path = os.path.join(student_dir, f"sample_{existing_samples+1}.png")
                        
                        # Save raw crop
                        cropped = crop_face(img_bgr, (x, y, w, h))
                        cv2.imwrite(sample_path, cropped)
                        
                        # 1. Register student in SQLite if not already registered in DB
                        is_registered = False
                        existing_student_db = get_student(student_id)
                        if not existing_student_db:
                            success = register_student(
                                student_id=student_id,
                                name=name,
                                roll_number=roll_number,
                                section=section,
                                department=department,
                                email=email
                            )
                            if success:
                                is_registered = True
                                st.write("🎉 Student registered in database.")
                            else:
                                st.error("❌ Database insertion failed. Roll number or Student ID may already exist.")
                                return
                        else:
                            is_registered = True
                            
                        if is_registered:
                            # 2. Extract Embedding and save in database
                            model = get_face_model(model_type)
                            preprocessed = normalize_image(resize_image(cropped, IMAGE_SIZE))
                            emb = get_embedding(model, preprocessed)
                            save_embedding(student_id, emb)
                            
                            # Force reload face recognition cache
                            from src.face_recognition import load_known_face_embeddings
                            load_known_face_embeddings(force_reload=True)
                            
                            st.success(f"💾 Saved sample {existing_samples+1}. Enrollment embedding written to database.")
                            st.info(f"You have registered {existing_samples+1} sample(s) for this student. We recommend capturing 5-10 samples in different angles for high accuracy.")
                            
    # Tab 2: OpenCV local webcam capture
    with tab2:
        st.write("For running locally. Clicking this button will open your local webcam, capture 15 frames automatically with different facial expressions/angles, and enroll them.")
        
        if demo_mode:
            st.warning("⚠️ Demo mode is active. Webcam actions are simulated.")
            if st.button("Simulate Automatic Enrollment"):
                # Register student
                success = register_student(
                    student_id=student_id,
                    name=name,
                    roll_number=roll_number,
                    section=section,
                    department=department,
                    email=email
                )
                if success:
                    os.makedirs(student_dir, exist_ok=True)
                    # Create mock embedding
                    model = get_face_model(model_type)
                    mock_face = np.random.uniform(0, 255, (IMAGE_SIZE[0], IMAGE_SIZE[1], 3)).astype(np.uint8)
                    cv2.imwrite(os.path.join(student_dir, "sample_1.png"), mock_face)
                    preprocessed = normalize_image(mock_face)
                    emb = get_embedding(model, preprocessed)
                    save_embedding(student_id, emb)
                    
                    from src.face_recognition import load_known_face_embeddings
                    load_known_face_embeddings(force_reload=True)
                    
                    st.success(f"🎉 [DEMO] Successfully registered student {name} with simulated biometric signatures.")
                else:
                    st.error("❌ Registration failed. Make sure student ID or Roll number is unique.")
        else:
            if st.button("Start Auto Webcam Capture (15 frames)"):
                cap = cv2.VideoCapture(0)
                if not cap.isOpened():
                    st.error("❌ OpenCV could not access your local webcam. Please check permissions or use the 'Browser Camera' tab.")
                    return
                    
                st.info("Webcam started! Please look at the camera, tilt your head slightly, and change expressions. Capturing 15 samples...")
                
                # Setup Streamlit placeholders
                frame_place = st.empty()
                progress_bar = st.progress(0)
                
                os.makedirs(student_dir, exist_ok=True)
                
                # First, register the student in DB
                db_success = register_student(
                    student_id=student_id,
                    name=name,
                    roll_number=roll_number,
                    section=section,
                    department=department,
                    email=email
                )
                
                if not db_success:
                    st.error("❌ Could not insert student details. Student ID or Roll number might be duplicate.")
                    cap.release()
                    return
                    
                captured_count = 0
                model = get_face_model(model_type)
                
                while captured_count < 15:
                    ret, frame = cap.read()
                    if not ret:
                        st.error("Failed to read webcam frame.")
                        break
                        
                    faces = detect_faces(frame)
                    display_frame = frame.copy()
                    
                    if len(faces) == 1:
                        x, y, w, h = faces[0]
                        # Quality check
                        is_valid, _ = check_face_quality(frame, (x, y, w, h))
                        
                        box_color = (0, 255, 0) if is_valid else (0, 165, 255)
                        cv2.rectangle(display_frame, (x, y), (x+w, y+h), box_color, 2)
                        
                        if is_valid:
                            captured_count += 1
                            sample_path = os.path.join(student_dir, f"sample_{captured_count}.png")
                            cropped = crop_face(frame, (x, y, w, h))
                            cv2.imwrite(sample_path, cropped)
                            
                            # Save embedding
                            preprocessed = normalize_image(resize_image(cropped, IMAGE_SIZE))
                            emb = get_embedding(model, preprocessed)
                            save_embedding(student_id, emb)
                            
                            # Update progress
                            progress_bar.progress(captured_count / 15)
                            
                    elif len(faces) > 1:
                        # Draw orange box on all faces
                        for (x, y, w, h) in faces:
                            cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 165, 255), 2)
                            
                    # Display the live feed with boxes in Streamlit
                    frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                    frame_place.image(frame_rgb, width=450)
                    time.sleep(0.1)  # Throttle to keep responsive
                    
                cap.release()
                frame_place.empty()
                
                # Reload face recognition cache
                from src.face_recognition import load_known_face_embeddings
                load_known_face_embeddings(force_reload=True)
                
                st.success(f"🎉 Registration and Enrollment complete! Captured 15 biometric samples for student: **{name}**.")
