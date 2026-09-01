import streamlit as st
import cv2
import numpy as np
import os
import sys
from PIL import Image
import pandas as pd
import time

# Add base directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import STUDENTS_DIR, IMAGE_SIZE
from src.database import (
    get_all_students, get_student, update_student,
    save_embedding, delete_student, delete_student_embeddings
)
from src.face_detection import detect_faces, crop_face
from src.preprocessing import check_face_quality, resize_image, normalize_image
from src.embeddings import get_face_model, get_embedding, sync_student_embeddings
from src.face_recognition import load_known_face_embeddings


def get_student_samples(student_id):
    """Returns a list of image filenames for the given student."""
    s_dir = os.path.join(STUDENTS_DIR, student_id)
    if not os.path.exists(s_dir):
        return []
    return [
        f for f in sorted(os.listdir(s_dir))
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]


def get_next_sample_filename(student_id):
    """Determines the next available sample filename for a student."""
    samples = get_student_samples(student_id)
    indices = []
    for f in samples:
        base = os.path.splitext(f)[0]
        if "_" in base:
            try:
                indices.append(int(base.split("_")[-1]))
            except ValueError:
                pass
    next_idx = (max(indices) + 1) if indices else (len(samples) + 1)
    return f"sample_{next_idx}.png"


def enroll_sample_image(student_id, img_bgr, bounding_box, active_model_type="mobilenet"):
    """
    Crops, saves the face sample to disk, extracts embeddings across models,
    and updates the face recognition cache.
    Returns (success: bool, sample_path: str, message: str).
    """
    s_dir = os.path.join(STUDENTS_DIR, student_id)
    os.makedirs(s_dir, exist_ok=True)

    cropped = crop_face(img_bgr, bounding_box)
    if cropped is None or cropped.size == 0:
        return False, None, "Failed to crop detected face region."

    filename = get_next_sample_filename(student_id)
    sample_path = os.path.join(s_dir, filename)
    cv2.imwrite(sample_path, cropped)

    # Generate and store embeddings for all supported models so model-switching always works
    preprocessed = normalize_image(resize_image(cropped, IMAGE_SIZE))
    for m_type in ["mobilenet", "custom_cnn", "custom_cnn_baseline"]:
        try:
            model = get_face_model(m_type)
            emb = get_embedding(model, preprocessed)
            if emb is not None:
                save_embedding(student_id, emb, model_name=m_type)
        except Exception:
            pass

    # Reload recognition cache
    load_known_face_embeddings(model_type=active_model_type, force_reload=True)
    return True, sample_path, f"Saved {filename} and enrolled biometrics."


def show(demo_mode=False, model_type="mobilenet"):
    st.title("👥 Student Database")
    st.markdown("### View student records, edit details, and manage biometric face samples")

    all_students = get_all_students()

    # KPI Metrics Header
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    total_students = len(all_students)
    total_samples = sum(len(get_student_samples(s["student_id"])) for s in all_students)
    avg_samples = (total_samples / total_students) if total_students > 0 else 0

    with col_m1:
        st.metric("Total Students", total_students)
    with col_m2:
        st.metric("Enrolled Face Samples", total_samples)
    with col_m3:
        st.metric("Avg. Samples / Student", f"{avg_samples:.1f}")
    with col_m4:
        st.metric("Active Biometric Model", model_type.upper())

    st.markdown("---")

    if not all_students:
        st.info("ℹ️ No students are registered in the system yet. Navigate to **👤 Register Student** in the sidebar to add your first student.")
        return

    # Use explicit conditional navigation instead of st.tabs so camera hardware is NEVER initialized unless Edit Photos is active
    sub_nav = st.radio(
        "Student Database Navigation",
        [
            "📋 Student Directory & Overview",
            "✏️ Edit Student Details",
            "📸 Edit Photos / Add Images"
        ],
        horizontal=True
    )

    st.markdown("---")

    # =========================================================================
    # SECTION 1: Student Directory & Overview (Zero camera code executed)
    # =========================================================================
    if sub_nav == "📋 Student Directory & Overview":
        st.subheader("📋 Registered Students Directory")

        # Filters
        f_col1, f_col2 = st.columns([1, 2])
        sections = sorted(list({s["section"] for s in all_students}))
        with f_col1:
            selected_section = st.selectbox("Filter by Section", ["All Sections"] + sections)
        with f_col2:
            search_query = st.text_input("🔍 Search by Name, Student ID, or Roll Number", placeholder="Type to search...").strip().lower()

        filtered_students = all_students
        if selected_section != "All Sections":
            filtered_students = [s for s in filtered_students if s["section"] == selected_section]
        if search_query:
            filtered_students = [
                s for s in filtered_students
                if search_query in s["name"].lower()
                or search_query in s["student_id"].lower()
                or search_query in s["roll_number"].lower()
            ]

        # Table Display
        table_rows = []
        for s in filtered_students:
            sample_count = len(get_student_samples(s["student_id"]))
            table_rows.append({
                "Student ID": s["student_id"],
                "Name": s["name"],
                "Roll Number": s["roll_number"],
                "Section": s["section"],
                "Department": s["department"],
                "Email": s["email"],
                "Enrolled Samples": sample_count,
                "Registered Date": s["created_at"]
            })

        if table_rows:
            st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)
        else:
            st.warning("No students matched your search filter.")

        st.markdown("---")
        st.subheader("🖼️ Student Face Sample Profiles")

        for s in filtered_students:
            sid = s["student_id"]
            samples = get_student_samples(sid)
            s_dir = os.path.join(STUDENTS_DIR, sid)

            with st.expander(f"👤 **{s['name']}** ({sid}) — {s['section']} | {len(samples)} face sample(s)"):
                col_info, col_gallery = st.columns([1, 2])
                with col_info:
                    st.markdown(f"**Roll Number:** `{s['roll_number']}`")
                    st.markdown(f"**Department:** {s['department']}")
                    st.markdown(f"**Email:** `{s['email']}`")
                    st.markdown(f"**Biometric Status:** {'✅ Enrolled (' + str(len(samples)) + ' samples)' if samples else '⚠️ No samples'}")

                with col_gallery:
                    if samples:
                        gallery_cols = st.columns(min(len(samples), 6))
                        for idx, sfile in enumerate(samples[:6]):
                            img_p = os.path.join(s_dir, sfile)
                            if os.path.exists(img_p):
                                with gallery_cols[idx % len(gallery_cols)]:
                                    st.image(Image.open(img_p), caption=sfile, use_container_width=True)
                        if len(samples) > 6:
                            st.caption(f"*...plus {len(samples) - 6} additional samples*")
                    else:
                        st.info("No face samples enrolled. Switch to **📸 Edit Photos / Add Images** above to add photos.")

    # =========================================================================
    # SECTION 2: Edit Student Details (Zero camera code executed)
    # =========================================================================
    elif sub_nav == "✏️ Edit Student Details":
        st.subheader("✏️ Edit Student Information")
        st.markdown("Select a student below to modify their registered credentials in the database.")

        student_options = {
            f"{s['name']} ({s['student_id']}) - Roll: {s['roll_number']}": s["student_id"]
            for s in all_students
        }
        selected_option = st.selectbox("Select Student to Edit", list(student_options.keys()), key="edit_student_selector")
        target_sid = student_options[selected_option]
        current_student = get_student(target_sid)

        if current_student:
            with st.form("edit_student_form"):
                st.write(f"Editing Student ID: **{target_sid}**")
                e_col1, e_col2 = st.columns(2)
                with e_col1:
                    new_name = st.text_input("Full Name", value=current_student["name"]).strip()
                    new_roll = st.text_input("Roll Number", value=current_student["roll_number"]).strip()
                    new_section = st.text_input("Class / Section", value=current_student["section"]).strip()
                with e_col2:
                    new_dept = st.text_input("Department", value=current_student["department"]).strip()
                    new_email = st.text_input("Email Address", value=current_student["email"]).strip()

                st.caption("Note: Student ID is the permanent primary key and cannot be altered.")
                submitted = st.form_submit_button("💾 Save Updated Details", type="primary")

                if submitted:
                    if not (new_name and new_roll and new_section and new_dept and new_email):
                        st.error("❌ All fields are required. Please fill in all information.")
                    else:
                        success, message = update_student(
                            student_id=target_sid,
                            name=new_name,
                            roll_number=new_roll,
                            section=new_section,
                            department=new_dept,
                            email=new_email
                        )
                        if success:
                            st.success(f"✅ {message}")
                            time.sleep(0.5)
                            st.rerun()
                        else:
                            st.error(f"❌ Update failed: {message}")

    # =========================================================================
    # SECTION 3: Edit Photos / Add Images (Only executed when explicitly selected)
    # =========================================================================
    elif sub_nav == "📸 Edit Photos / Add Images":
        st.subheader("📸 Edit Photos & Add Face Samples")
        st.markdown("Manage existing facial photos or enroll additional samples for any student.")

        add_student_options = {
            f"{s['name']} ({s['student_id']})": s["student_id"]
            for s in all_students
        }
        selected_add_option = st.selectbox("Select Student", list(add_student_options.keys()), key="add_samples_student_selector")
        add_sid = add_student_options[selected_add_option]
        student_info = get_student(add_sid)

        current_samples = get_student_samples(add_sid)
        st.markdown(f"Student: **{student_info['name']}** ({add_sid}) | Enrolled Samples: **{len(current_samples)}**")

        # 1. Manage & Delete Existing Samples
        if current_samples:
            st.write("#### Enrolled Samples Gallery & Management")
            s_dir = os.path.join(STUDENTS_DIR, add_sid)
            cols = st.columns(min(len(current_samples), 4))
            for i, sample_file in enumerate(current_samples):
                img_path = os.path.join(s_dir, sample_file)
                if os.path.exists(img_path):
                    with cols[i % len(cols)]:
                        st.image(Image.open(img_path), caption=sample_file, use_container_width=True)
                        if st.button(f"🗑️ Delete", key=f"del_sample_{add_sid}_{sample_file}", help=f"Delete {sample_file}"):
                            try:
                                os.remove(img_path)
                                # Re-sync embeddings for this student from remaining samples
                                delete_student_embeddings(add_sid)
                                sync_student_embeddings(model_type)
                                load_known_face_embeddings(model_type=model_type, force_reload=True)
                                st.success(f"Deleted {sample_file}.")
                                time.sleep(0.5)
                                st.rerun()
                            except Exception as del_err:
                                st.error(f"Error deleting file: {del_err}")

        st.markdown("---")
        st.write("#### Add New Face Images:")

        # Default to Upload Image Files so camera hardware is never touched until user explicitly activates camera
        capture_method = st.radio(
            "Choose Image Source",
            ["📁 Upload Image Files", "📸 Browser Camera", "🖥️ Local Webcam / Demo Mode"],
            horizontal=True
        )

        # Method 1: File Upload (No camera hardware needed)
        if capture_method == "📁 Upload Image Files":
            st.write("Upload face photos (`.png`, `.jpg`, `.jpeg`) from your device.")
            uploaded_files = st.file_uploader(
                "Select images to enroll",
                type=["png", "jpg", "jpeg"],
                accept_multiple_files=True,
                key=f"uploader_add_{add_sid}"
            )

            if uploaded_files:
                st.write(f"Selected **{len(uploaded_files)}** file(s).")
                if st.button("🚀 Process & Enroll Uploaded Images", type="primary", key="btn_process_uploads"):
                    progress = st.progress(0.0)
                    success_count = 0
                    failures = []

                    for idx, uploaded_file in enumerate(uploaded_files):
                        try:
                            pil_img = Image.open(uploaded_file)
                            img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

                            faces = detect_faces(img_bgr)
                            if len(faces) == 0:
                                failures.append(f"{uploaded_file.name}: No face detected.")
                            elif len(faces) > 1:
                                failures.append(f"{uploaded_file.name}: Multiple faces detected ({len(faces)}).")
                            else:
                                x, y, w, h = faces[0]
                                ok, p, m = enroll_sample_image(add_sid, img_bgr, (x, y, w, h), active_model_type=model_type)
                                if ok:
                                    success_count += 1
                                else:
                                    failures.append(f"{uploaded_file.name}: {m}")
                        except Exception as file_err:
                            failures.append(f"{uploaded_file.name}: Error processing image ({file_err}).")

                        progress.progress((idx + 1) / len(uploaded_files))

                    if success_count > 0:
                        st.success(f"🎉 Successfully enrolled **{success_count}** new sample(s) for **{student_info['name']}**!")
                    if failures:
                        st.warning("⚠️ Some files could not be enrolled:")
                        for fail in failures:
                            st.caption(f"- {fail}")
                    time.sleep(1.0)
                    st.rerun()

        # Method 2: Browser Camera (Only requests camera when explicit checkbox is checked)
        elif capture_method == "📸 Browser Camera":
            st.write("Use your browser's camera to take a photo.")
            camera_active = st.checkbox("📷 Turn On Camera to Take Photo", value=False, key=f"cam_active_{add_sid}")

            if camera_active:
                camera_img = st.camera_input("Take Photo", key=f"camera_add_{add_sid}")

                if camera_img is not None:
                    pil_img = Image.open(camera_img)
                    img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

                    faces = detect_faces(img_bgr)
                    if len(faces) == 0:
                        st.error("❌ No face detected. Please ensure your face is well-lit and fully visible.")
                    elif len(faces) > 1:
                        st.error(f"❌ Multiple faces ({len(faces)}) detected. Only one person should be in the frame.")
                    else:
                        x, y, w, h = faces[0]
                        is_valid, reason = check_face_quality(img_bgr, (x, y, w, h))

                        feedback_img = img_bgr.copy()
                        box_color = (0, 255, 0) if is_valid else (0, 165, 255)
                        cv2.rectangle(feedback_img, (x, y), (x+w, y+h), box_color, 2)
                        st.image(cv2.cvtColor(feedback_img, cv2.COLOR_BGR2RGB), caption="Face Detection Feedback", width=280)

                        if not is_valid:
                            st.warning(f"⚠️ Quality check notice: {reason}")

                        if st.button("💾 Enroll this Sample", type="primary", key="btn_save_browser_cam"):
                            success, path, msg = enroll_sample_image(add_sid, img_bgr, (x, y, w, h), active_model_type=model_type)
                            if success:
                                st.success(f"🎉 {msg}")
                                time.sleep(0.6)
                                st.rerun()
                            else:
                                st.error(f"❌ {msg}")
            else:
                st.info("👆 Check **'Turn On Camera to Take Photo'** above when you are ready to take a snapshot.")

        # Method 3: Local Webcam / Demo Mode Capture
        elif capture_method == "🖥️ Local Webcam / Demo Mode":
            st.write("Batch capture multiple face samples from your local webcam to cover multiple head tilts and angles.")

            num_frames = st.slider("Number of samples to capture", min_value=3, max_value=15, value=5)

            if demo_mode:
                st.info("🔬 Demo mode is active. Webcam frames will be simulated.")
                if st.button(f"Simulate Batch Enrollment ({num_frames} samples)", key="btn_sim_add"):
                    s_dir = os.path.join(STUDENTS_DIR, add_sid)
                    os.makedirs(s_dir, exist_ok=True)
                    for _ in range(num_frames):
                        mock_face = np.random.uniform(0, 255, (IMAGE_SIZE[0], IMAGE_SIZE[1], 3)).astype(np.uint8)
                        fname = get_next_sample_filename(add_sid)
                        cv2.imwrite(os.path.join(s_dir, fname), mock_face)
                        preprocessed = normalize_image(mock_face)
                        for m_type in ["mobilenet", "custom_cnn", "custom_cnn_baseline"]:
                            model = get_face_model(m_type)
                            emb = get_embedding(model, preprocessed)
                            if emb is not None:
                                save_embedding(add_sid, emb, model_name=m_type)
                    load_known_face_embeddings(model_type=model_type, force_reload=True)
                    st.success(f"🎉 [DEMO] Successfully added {num_frames} simulated samples for {student_info['name']}.")
                    time.sleep(0.8)
                    st.rerun()
            else:
                if st.button(f"Start Webcam Capture ({num_frames} frames)", key="btn_webcam_add"):
                    cap = cv2.VideoCapture(0)
                    if not cap.isOpened():
                        st.error("❌ Local webcam not accessible. Use the 'Upload Image Files' or 'Browser Camera' option above.")
                    else:
                        frame_place = st.empty()
                        progress_bar = st.progress(0.0)
                        captured = 0

                        while captured < num_frames:
                            ret, frame = cap.read()
                            if not ret:
                                break

                            faces = detect_faces(frame)
                            display_f = frame.copy()

                            if len(faces) == 1:
                                x, y, w, h = faces[0]
                                is_valid, _ = check_face_quality(frame, (x, y, w, h))
                                color = (0, 255, 0) if is_valid else (0, 165, 255)
                                cv2.rectangle(display_f, (x, y), (x+w, y+h), color, 2)

                                if is_valid:
                                    captured += 1
                                    enroll_sample_image(add_sid, frame, (x, y, w, h), active_model_type=model_type)
                                    progress_bar.progress(captured / num_frames)
                                    time.sleep(0.15)
                            elif len(faces) > 1:
                                for (x, y, w, h) in faces:
                                    cv2.rectangle(display_f, (x, y), (x+w, y+h), (0, 165, 255), 2)

                            frame_place.image(cv2.cvtColor(display_f, cv2.COLOR_BGR2RGB), width=420)
                            time.sleep(0.05)

                        cap.release()
                        frame_place.empty()
                        st.success(f"🎉 Enrolled {captured} new biometric sample(s) for **{student_info['name']}**!")
                        time.sleep(0.8)
                        st.rerun()
