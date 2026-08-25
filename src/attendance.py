import os
import sys
from datetime import datetime

# Add base directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.database import mark_attendance, is_attendance_marked, log_recognition

def process_attendance_event(student_id, confidence, session="Default"):
    """
    Processes a face recognition event. If the student is known,
    checks for duplicates, logs the event, and marks attendance.
    Args:
        student_id: Registered student ID or "Unknown"
        confidence: Recognition confidence score (0.0 to 1.0)
        session: Attendance session (e.g., "Default", "Morning", "Afternoon")
    Returns:
        is_marked: bool (True if a new attendance record was written)
        message: str user-facing status message
    """
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")
    
    # 1. Log the recognition attempt (for auditing & Course 3 evaluation statistics)
    log_status = "Success" if student_id != "Unknown" else "Failure"
    log_recognition(student_id if student_id != "Unknown" else None, confidence, log_status)
    
    if student_id == "Unknown":
        return False, "Unknown face detected. Attendance not marked."
        
    # 2. Check for duplicate attendance in the current session
    if is_attendance_marked(student_id, date_str, session):
        return False, "Attendance already marked."
        
    # 3. Mark attendance in database
    success, message = mark_attendance(
        student_id=student_id,
        date_str=date_str,
        time_str=time_str,
        session_str=session,
        confidence=confidence,
        status="Present"
    )
    
    return success, message
