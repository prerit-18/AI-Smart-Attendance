import streamlit as st
import cv2
import numpy as np
import os
import sys
from datetime import datetime
import time

# Add base directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import RECOGNITION_THRESHOLD, IMAGE_SIZE
from src.database import get_all_students, get_student
from src.face_detection import detect_faces, crop_face
from src.preprocessing import resize_image, normalize_image
from src.embeddings import get_face_model, get_embedding
from src.face_recognition import recognize_face, load_known_face_embeddings
from src.attendance import process_attendance_event

def show(demo_mode=False, model_type="custom_cnn"):
    st.title("🎥 Live Attendance Session")
    st.markdown("### Deep Learning Real-Time Face Recognition & Auto Attendance")
    
    # Reload known face embeddings cache
    load_known_face_embeddings(force_reload=True)
    
    # Configure session parameters
    session_name = st.sidebar.text_input("Current Session Name", value="Default Class").strip()
    rec_threshold = st.sidebar.slider(
        "Face Similarity Threshold", 
        min_value=0.0, max_value=1.0, 
        value=RECOGNITION_THRESHOLD, step=0.05,
        help="Higher values make recognition stricter (fewer false positives, more false negatives)."
    )
    
    # Fetch registered students for reference
    registered_students = get_all_students()
    
    if not registered_students:
        st.warning("⚠️ No students registered in the system yet. Please go to 'Register Student' first.")
        return
        
    st.write(f"Active Session: **{session_name}** | Threshold: **{rec_threshold}** | Active Model: **{model_type}**")
    
    # Demo Mode simulation section
    if demo_mode:
        st.info("🔬 **Demo Mode Active**: You can click the button below to simulate random attendance check-ins for database testing, without using a physical camera.")
        if st.button("🎲 Simulate Random Student Attendance"):
            import random
            selected_student = random.choice(registered_students)
            confidence = random.uniform(0.75, 0.99)
            marked, msg = process_attendance_event(
                student_id=selected_student["student_id"],
                confidence=confidence,
                session=session_name
            )
            if marked:
                st.success(f"🔔 [SIMULATED] Marked **{selected_student['name']}** as PRESENT (Conf: {confidence*100:.1f}%).")
            else:
                st.info(f"💾 [SIMULATED] {selected_student['name']}: {msg}")
        st.markdown("---")

    # Create navigation tabs for Capture Method
    tab1, tab2 = st.tabs(["📸 Browser Camera (Web App standard)", "🖥️ Local Webcam (Direct Stream)"])
    
    # Tab 1: Browser Camera (using st.camera_input)
    with tab1:
        st.write("Take a snapshot using your web browser's camera to verify and mark attendance.")
        camera_img = st.camera_input("Position your face in the center of the frame", key="live_cam_input")
        
        if camera_img is not None:
            from PIL import Image
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
                
                # Crop and quality verification
                face_crop = crop_face(img_bgr, (x, y, w, h))
                if face_crop is not None and face_crop.size > 0:
                    model = get_face_model(model_type)
                    preprocessed = normalize_image(resize_image(face_crop, IMAGE_SIZE))
                    emb = get_embedding(model, preprocessed)
                    
                    # Run face recognition model
                    student_id, confidence, is_known = recognize_face(emb, threshold=rec_threshold)
                    
                    if is_known:
                        student_info = get_student(student_id)
                        
                        # Draw green box for matching face
                        feedback_img = img_bgr.copy()
                        cv2.rectangle(feedback_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        feedback_rgb = cv2.cvtColor(feedback_img, cv2.COLOR_BGR2RGB)
                        st.image(feedback_rgb, caption=f"Verified: {student_info['name']}", width=300)
                        
                        # Process and mark attendance
                        marked, msg = process_attendance_event(
                            student_id=student_id,
                            confidence=confidence,
                            session=session_name
                        )
                        if marked:
                            st.success(f"🎉 Marked **{student_info['name']}** as PRESENT (Confidence: {confidence*100:.1f}%).")
                        else:
                            st.info(f"💾 {student_info['name']}: {msg}")
                    else:
                        # Draw red box for unknown face
                        feedback_img = img_bgr.copy()
                        cv2.rectangle(feedback_img, (x, y), (x+w, y+h), (0, 0, 255), 2)
                        feedback_rgb = cv2.cvtColor(feedback_img, cv2.COLOR_BGR2RGB)
                        st.image(feedback_rgb, caption="Unknown Face", width=300)
                        st.error("❌ Identity not verified. Face did not match any registered student.")

    # Tab 2: Direct Local Webcam (using OpenCV video capture stream)
    with tab2:
        st.write("Start a live, continuous video stream to automatically detect, verify, and mark attendance.")
        
        run_camera = st.checkbox("📸 Start Live Webcam Stream")
        video_place = st.empty()
        status_place = st.empty()
        
        if run_camera:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("❌ Local webcam not accessible. Ensure no other application is using it.")
                return
                
            model = get_face_model(model_type)
            frame_count = 0
            last_detections = []
            
            while run_camera:
                ret, frame = cap.read()
                if not ret:
                    st.warning("Failed to receive video frame.")
                    break
                    
                frame_count += 1
                
                # Process recognition every 3rd frame to optimize performance
                if frame_count % 3 == 0:
                    faces = detect_faces(frame)
                    
                    if len(faces) > 0:
                        new_detections = []
                        for (x, y, w, h) in faces:
                            face_crop = crop_face(frame, (x, y, w, h))
                            if face_crop is not None and face_crop.size > 0:
                                preprocessed = normalize_image(resize_image(face_crop, IMAGE_SIZE))
                                emb = get_embedding(model, preprocessed)
                                
                                # Recognize face
                                student_id, confidence, is_known = recognize_face(emb, threshold=rec_threshold)
                                
                                if is_known:
                                    # Mark attendance
                                    marked, msg = process_attendance_event(
                                        student_id=student_id,
                                        confidence=confidence,
                                        session=session_name
                                    )
                                    
                                    student_info = get_student(student_id)
                                    name_label = student_info['name'].upper()
                                    status_label = "PRESENT" if marked else "MARKED"
                                    box_color = (0, 255, 0)
                                    
                                    if marked:
                                        status_place.success(f"🔔 Marked **{student_info['name']}** as PRESENT (Conf: {confidence*100:.1f}%).")
                                else:
                                    # Log unknown face attempt
                                    process_attendance_event("Unknown", confidence, session=session_name)
                                    name_label = "UNKNOWN"
                                    status_label = ""
                                    box_color = (0, 0, 255)
                                    student_id = "Unknown"
                                    
                                new_detections.append({
                                    "bbox": (x, y, w, h),
                                    "student_id": student_id,
                                    "name": name_label,
                                    "status": status_label,
                                    "confidence": confidence,
                                    "color": box_color
                                })
                        
                        # Match and smooth coordinates with EMA to reduce flickering
                        updated_detections = []
                        for nd in new_detections:
                            nx, ny, nw, nh = nd["bbox"]
                            matched = False
                            for od in last_detections:
                                ox, oy, ow, oh = od["bbox"]
                                dist = np.sqrt(((nx + nw/2) - (ox + ow/2))**2 + ((ny + nh/2) - (oy + oh/2))**2)
                                if dist < 100:
                                    sx = int(0.7 * ox + 0.3 * nx)
                                    sy = int(0.7 * oy + 0.3 * ny)
                                    sw = int(0.7 * ow + 0.3 * nw)
                                    sh = int(0.7 * oh + 0.3 * nh)
                                    updated_detections.append({
                                        "bbox": (sx, sy, sw, sh),
                                        "student_id": nd["student_id"],
                                        "name": nd["name"],
                                        "status": nd["status"],
                                        "confidence": nd["confidence"],
                                        "color": nd["color"],
                                        "lifetime": 30
                                    })
                                    matched = True
                                    break
                            if not matched:
                                nd["lifetime"] = 30
                                updated_detections.append(nd)
                                
                        for od in last_detections:
                            ox, oy, ow, oh = od["bbox"]
                            already_matched = False
                            for ud in updated_detections:
                                ux, uy, uw, uh = ud["bbox"]
                                udist = np.sqrt(((ux + uw/2) - (ox + ow/2))**2 + ((uy + uh/2) - (oy + oh/2))**2)
                                if udist < 30:
                                    already_matched = True
                                    break
                            if not already_matched:
                                od["lifetime"] -= 3
                                if od["lifetime"] > 0:
                                    updated_detections.append(od)
                                    
                        last_detections = updated_detections
                    else:
                        active_detections = []
                        for det in last_detections:
                            det["lifetime"] -= 3
                            if det["lifetime"] > 0:
                                active_detections.append(det)
                        last_detections = active_detections
                else:
                    active_detections = []
                    for det in last_detections:
                        det["lifetime"] -= 1
                        if det["lifetime"] > 0:
                            active_detections.append(det)
                    last_detections = active_detections
                            
                # Draw boxes and render labels
                for det in last_detections:
                    x, y, w, h = det["bbox"]
                    cv2.rectangle(frame, (x, y), (x+w, y+h), det["color"], 2)
                    
                    if det["student_id"] == "Unknown":
                        label = f"UNKNOWN ({det['confidence']*100:.1f}%)"
                        font_scale = 1.0
                    else:
                        label = f"{det['name']} [ID: {det['student_id']}]"
                        font_scale = 1.1
                        
                    (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 3)
                    banner_w = max(w, text_w + 16)
                    banner_x = max(0, min(x - (banner_w - w) // 2, frame.shape[1] - banner_w))
                    banner_y = max(80, y)
                    
                    cv2.rectangle(frame, (banner_x, banner_y - 80), (banner_x + banner_w, banner_y), (255, 255, 255), -1)
                    cv2.rectangle(frame, (banner_x, banner_y - 80), (banner_x + banner_w, banner_y), det["color"], 3)
                    
                    if det["student_id"] == "Unknown":
                        cv2.putText(frame, label, (banner_x + 8, banner_y - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 3)
                    else:
                        cv2.putText(frame, label, (banner_x + 8, banner_y - 45), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 3)
                        cv2.putText(frame, det["status"], (banner_x + 8, banner_y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                        
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                video_place.image(frame_rgb, channels="RGB", width=640)
                time.sleep(0.03)
                
            cap.release()
            video_place.empty()
            status_place.empty()
