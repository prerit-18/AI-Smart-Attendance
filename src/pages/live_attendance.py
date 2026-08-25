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

def show(demo_mode=False):
    st.title("🎥 Live Attendance Session")
    st.markdown("### Deep Learning Real-Time Face Recognition & Auto Attendance")
    
    # Reload known face embeddings cache
    load_known_face_embeddings(force_reload=True)
    
    # Configure session name
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
        
    st.write(f"Active Session: **{session_name}** | Threshold: **{rec_threshold}**")
    
    # Demo Mode UI
    if demo_mode:
        st.info("ℹ️ **DEMO MODE ACTIVE**: You can simulate face detection events on the camera by mapping any detected face to a selected registered student profile.")
        
        # 1. Selection dropdown
        student_options = {f"{s['name']} ({s['student_id']})": s['student_id'] for s in registered_students}
        student_options["Unknown Person (Intruder)"] = "Unknown"
        
        selected_option = st.selectbox("Select Student Profile to Force Simulate", list(student_options.keys()))
        simulated_student_id = student_options[selected_option]
        
        # 2. Camera Trigger
        run_camera = st.checkbox("📸 Start Simulation Webcam Feed")
        
        video_place = st.empty()
        status_place = st.empty()
        
        if run_camera:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("❌ Local webcam not accessible. Ensure camera permission is granted.")
                return
                
            frame_count = 0
            last_detections = []
            
            while run_camera:
                ret, frame = cap.read()
                if not ret:
                    st.warning("Failed to receive video frame.")
                    break
                    
                frame_count += 1
                
                # Perform face detection on every 3rd frame
                if frame_count % 3 == 0:
                    faces = detect_faces(frame, scaleFactor=1.05, minNeighbors=4)
                    
                    if len(faces) > 0:
                        new_detections = []
                        for (x, y, w, h) in faces:
                            if simulated_student_id == "Unknown":
                                # Log failed attempt in DB
                                process_attendance_event("Unknown", 0.35, session=session_name)
                                new_detections.append({
                                    "bbox": (x, y, w, h),
                                    "student_id": "Unknown",
                                    "name": "UNKNOWN",
                                    "status": "",
                                    "confidence": 0.35,
                                    "color": (0, 0, 255)
                                })
                            else:
                                # Fetch details of target student
                                student_info = get_student(simulated_student_id)
                                name_label = student_info['name'].upper()
                                confidence = 0.98
                                
                                # Mark attendance in database
                                marked, msg = process_attendance_event(
                                    student_id=simulated_student_id,
                                    confidence=confidence,
                                    session=session_name
                                )
                                
                                status_label = "PRESENT" if marked else "MARKED"
                                new_detections.append({
                                    "bbox": (x, y, w, h),
                                    "student_id": simulated_student_id,
                                    "name": name_label,
                                    "status": status_label,
                                    "confidence": confidence,
                                    "color": (0, 255, 0)
                                })
                                
                                if marked:
                                    status_place.success(f"🔔 [DEMO] Marked **{student_info['name']}** as PRESENT.")
                                else:
                                    status_place.info(f"💾 [DEMO] {student_info['name']}: {msg}")
                                    
                        # Match and smooth new detections with cached detections (using EMA)
                        updated_detections = []
                        for nd in new_detections:
                            nx, ny, nw, nh = nd["bbox"]
                            matched = False
                            for od in last_detections:
                                ox, oy, ow, oh = od["bbox"]
                                dist = np.sqrt(((nx + nw/2) - (ox + ow/2))**2 + ((ny + nh/2) - (oy + oh/2))**2)
                                if dist < 100:
                                    # Smooth coordinates (0.7 old + 0.3 new) to prevent jitter
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
                                
                        # Keep unmatched old detections but decay lifetime (3 frames elapsed)
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
                        # Decay lifetime of cached boxes (3 frames elapsed)
                        active_detections = []
                        for det in last_detections:
                            det["lifetime"] -= 3
                            if det["lifetime"] > 0:
                                active_detections.append(det)
                        last_detections = active_detections
                else:
                    # Decay lifetime on every intermediate frame (1 frame elapsed)
                    active_detections = []
                    for det in last_detections:
                        det["lifetime"] -= 1
                        if det["lifetime"] > 0:
                            active_detections.append(det)
                    last_detections = active_detections
                        
                # Draw the cached boxes on EVERY frame to prevent flickering!
                for det in last_detections:
                    x, y, w, h = det["bbox"]
                    cv2.rectangle(frame, (x, y), (x+w, y+h), det["color"], 2)
                    
                    # Prepare label text and font scale (Doubled sizes: scale 1.1 / 1.4, thickness 3)
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
                    
                    # Render text inside the banner
                    if det["student_id"] == "Unknown":
                        cv2.putText(frame, label, (banner_x + 8, banner_y - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 3)
                    else:
                        cv2.putText(frame, label, (banner_x + 8, banner_y - 45), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 3)
                        cv2.putText(frame, det["status"], (banner_x + 8, banner_y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                        
                # Render the webcam stream
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                video_place.image(frame_rgb, channels="RGB", width=640)
                time.sleep(0.03)
                
            cap.release()
            video_place.empty()
            status_place.empty()
                        
    # Live Camera Feed UI
    else:
        # Camera Feed Loop control
        run_camera = st.checkbox("📸 Start Live Webcam Stream")
        
        # Streamlit empty placeholders for video rendering
        video_place = st.empty()
        status_place = st.empty()
        
        if run_camera:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("❌ Local webcam not accessible. Ensure no other application is using it, or enable 'Demo Mode' in the sidebar.")
                return
                
            model = get_face_model("mobilenet")
            
            # Timing variables to prevent high CPU loads (process every 3rd frame)
            frame_count = 0
            
            # Caching variable for rendering boxes continuously without flickering
            last_detections = []
            
            while run_camera:
                ret, frame = cap.read()
                if not ret:
                    st.warning("Failed to receive video frame.")
                    break
                    
                frame_count += 1
                
                # Perform face detection and classification on every 3rd frame
                if frame_count % 3 == 0:
                    faces = detect_faces(frame, scaleFactor=1.05, minNeighbors=4)
                    
                    if len(faces) > 0:
                        new_detections = []
                        for (x, y, w, h) in faces:
                            # 1. Quality Check
                            face_crop = crop_face(frame, (x, y, w, h))
                            if face_crop is not None and face_crop.size > 0:
                                # 2. Extract Embedding (normalize_image now converts BGR to RGB automatically!)
                                preprocessed = normalize_image(resize_image(face_crop, IMAGE_SIZE))
                                emb = get_embedding(model, preprocessed)
                                
                                # 3. Classify similarity
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
                                    box_color = (0, 255, 0) # Green
                                    
                                    if marked:
                                        status_place.success(f"🔔 Marked **{student_info['name']}** as PRESENT.")
                                else:
                                    # Log unknown face attempt (Failure)
                                    process_attendance_event("Unknown", confidence, session=session_name)
                                    name_label = "UNKNOWN"
                                    status_label = ""
                                    box_color = (0, 0, 255) # Red
                                    student_id = "Unknown"
                                    
                                new_detections.append({
                                    "bbox": (x, y, w, h),
                                    "student_id": student_id,
                                    "name": name_label,
                                    "status": status_label,
                                    "confidence": confidence,
                                    "color": box_color
                                })
                        
                        # Match and smooth new detections with cached detections (using EMA)
                        updated_detections = []
                        for nd in new_detections:
                            nx, ny, nw, nh = nd["bbox"]
                            matched = False
                            for od in last_detections:
                                ox, oy, ow, oh = od["bbox"]
                                dist = np.sqrt(((nx + nw/2) - (ox + ow/2))**2 + ((ny + nh/2) - (oy + oh/2))**2)
                                if dist < 100:
                                    # Smooth coordinates (0.7 old + 0.3 new) to prevent jitter
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
                                
                        # Keep unmatched old detections but decay lifetime (3 frames elapsed)
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
                        # Decay lifetime of cached boxes (3 frames elapsed)
                        active_detections = []
                        for det in last_detections:
                            det["lifetime"] -= 3
                            if det["lifetime"] > 0:
                                active_detections.append(det)
                        last_detections = active_detections
                else:
                    # Decay lifetime on every intermediate frame (1 frame elapsed)
                    active_detections = []
                    for det in last_detections:
                        det["lifetime"] -= 1
                        if det["lifetime"] > 0:
                            active_detections.append(det)
                    last_detections = active_detections
                            
                # Draw the cached boxes on EVERY frame to prevent flickering!
                for det in last_detections:
                    x, y, w, h = det["bbox"]
                    cv2.rectangle(frame, (x, y), (x+w, y+h), det["color"], 2)
                    
                    # Prepare label text and font scale (Doubled sizes: scale 1.1 / 1.4, thickness 3)
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
                    
                    # Render text inside the banner
                    if det["student_id"] == "Unknown":
                        cv2.putText(frame, label, (banner_x + 8, banner_y - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 3)
                    else:
                        cv2.putText(frame, label, (banner_x + 8, banner_y - 45), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 3)
                        cv2.putText(frame, det["status"], (banner_x + 8, banner_y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                        
                # Convert colors and show in Streamlit
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                video_place.image(frame_rgb, channels="RGB", width=640)
                
                # Check if checkbox was unchecked during loop execution
                time.sleep(0.03)  # Throttle frame rate (~30fps) to keep streamlit UI responsive
                
            cap.release()
            video_place.empty()
            status_place.empty()
