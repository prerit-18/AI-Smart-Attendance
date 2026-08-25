import streamlit as st
import pandas as pd
import os
import sys
import shutil

# Add base directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.database import get_attendance_records, get_all_students, delete_student, delete_attendance_record
from config.config import STUDENTS_DIR, DATABASE_PATH

def show():
    st.title("📋 Attendance Records & Management")
    st.markdown("### View logs, export spreadsheets, and manage database registers")
    
    # Simple navigation tabs: Records Viewer vs. Student Manager
    tab_records, tab_manage = st.tabs(["🔍 View Attendance Logs", "⚙️ Manage Registered Students"])
    
    with tab_records:
        st.subheader("Filter and Search Attendance")
        
        # 1. Filters block
        col1, col2, col3, col4 = st.columns(4)
        
        # Fetch sections and students dynamically for filter lists
        all_students = get_all_students()
        sections_list = list(set([s['section'] for s in all_students])) if all_students else []
        
        with col1:
            date_filter = st.date_input("Filter by Date", value=None)
            date_str = date_filter.strftime("%Y-%m-%d") if date_filter else None
        with col2:
            student_search = st.text_input("Search Student ID/Name", "").strip()
        with col3:
            selected_section = st.selectbox("Filter by Section", ["All"] + sections_list)
            section_str = None if selected_section == "All" else selected_section
        with col4:
            selected_status = st.selectbox("Filter by Status", ["All", "Present", "Absent", "Late"])
            status_str = None if selected_status == "All" else selected_status
            
        # 2. Query and Display
        records = get_attendance_records(
            date_filter=date_str,
            student_filter=student_search if student_search else None,
            section_filter=section_str,
            status_filter=status_str
        )
        
        if records:
            df = pd.DataFrame(records)
            # Reorder columns for display
            display_cols = ["student_id", "name", "roll_number", "section", "department", "date", "time", "session", "confidence", "status"]
            df_display = df[display_cols]
            
            # Format confidence to percent
            df_display["confidence"] = df_display["confidence"].apply(lambda x: f"{x*100:.1f}%")
            
            st.dataframe(df_display, use_container_width=True)
            
            # 3. CSV Export
            # Generate CSV bytes
            csv_data = df_display.to_csv(index=False).encode('utf-8')
            
            st.download_button(
                label="📥 Download Attendance CSV",
                data=csv_data,
                file_name=f"attendance_export_{datetime_str_safe()}.csv",
                mime="text/csv",
                help="Download this filtered view as a CSV spreadsheet."
            )
            
            # 4. Individual Attendance Log Deletion Form
            st.markdown("---")
            st.write("#### 🗑️ Remove Attendance Record")
            
            # Format record option keys linked to their row ID
            record_options = {
                f"{r['date']} {r['time']} - {r['name']} ({r['student_id']}) [{r['session']}]": r['id']
                for r in records
            }
            
            with st.form("delete_record_form"):
                selected_record = st.selectbox(
                    "Select Attendance Entry to Delete",
                    options=list(record_options.keys())
                )
                confirm_del = st.checkbox("Confirm you want to permanently delete this check-in record.")
                submit_del = st.form_submit_button("Delete Attendance Log", type="primary")
                
                if submit_del:
                    if not confirm_del:
                        st.error("Please check the confirmation box to delete the log.")
                    else:
                        target_id = record_options[selected_record]
                        if delete_attendance_record(target_id):
                            st.success("Attendance record successfully removed from database.")
                            st.rerun()
                        else:
                            st.error("Error occurred while deleting record.")
        else:
            st.info("No attendance records match the selected filters.")
            
    with tab_manage:
        st.subheader("Registered Student Biometric Directory")
        st.markdown("⚠️ **Caution**: Deleting a student permanently erases their biometric face embeddings, captured image samples, and past attendance logs.")
        
        if not all_students:
            st.info("No students registered.")
        else:
            # Table of students
            df_students = pd.DataFrame(all_students)
            display_student_cols = ["student_id", "name", "roll_number", "section", "department", "email"]
            st.dataframe(df_students[display_student_cols], use_container_width=True)
            
            # Deletion flow
            with st.form("delete_student_form"):
                st.write("#### Remove Student Biometrics")
                student_to_delete = st.selectbox("Select Student to Delete", [f"{s['name']} ({s['student_id']})" for s in all_students])
                confirm_check = st.checkbox("I confirm I want to permanently delete this student and all corresponding data.")
                
                delete_btn = st.form_submit_button("Delete Registered Student", type="primary")
                
                if delete_btn:
                    if not confirm_check:
                        st.error("Please select the confirmation checkbox to authorize deletion.")
                    else:
                        # Extract student ID from selection text "Name (STxxx)"
                        sid = student_to_delete.split("(")[-1].replace(")", "").strip()
                        
                        # 1. Delete student from SQLite
                        db_removed = delete_student(sid)
                        
                        # 2. Delete face samples directory
                        folder_path = os.path.join(STUDENTS_DIR, sid)
                        dir_removed = False
                        if os.path.exists(folder_path):
                            try:
                                shutil.rmtree(folder_path)
                                dir_removed = True
                            except Exception as e:
                                st.error(f"Error removing image samples folder: {e}")
                                
                        # 3. Force reload face recognition cache
                        from src.face_recognition import load_known_face_embeddings
                        load_known_face_embeddings(force_reload=True)
                        
                        if db_removed:
                            st.success(f"🗑️ Successfully deleted student **{sid}** from database.")
                            if dir_removed:
                                st.write("✅ Cleaned up captured biometric image files on disk.")
                            st.rerun()
                        else:
                            st.error(f"Failed to delete student {sid}.")

def datetime_str_safe():
    from datetime import datetime
    return datetime.now().strftime("%Y%m%d_%H%M%S")
