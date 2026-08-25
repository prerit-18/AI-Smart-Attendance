import pytest
import os
import sys

# Add base directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config.config as config
from src.database import init_db, register_student, get_attendance_records
from src.attendance import process_attendance_event

@pytest.fixture(autouse=True)
def setup_test_db(request):
    test_name = request.node.name
    test_db_path = os.path.join(config.DB_DIR, f"test_db_{test_name}.db")
    config.DATABASE_PATH = test_db_path
    
    if os.path.exists(test_db_path):
        try:
            os.remove(test_db_path)
        except Exception:
            pass
            
    init_db()
    # Register dummy student
    register_student(
        student_id="ST777",
        name="Attendance Tester",
        roll_number="777000",
        section="TEST-B",
        department="QA",
        email="tester@qa.com"
    )
    yield
    if os.path.exists(test_db_path):
        try:
            os.remove(test_db_path)
        except Exception:
            pass

def test_attendance_marking_and_duplicates():
    # 1. Mark attendance for registered student
    success, msg = process_attendance_event(
        student_id="ST777",
        confidence=0.88,
        session="Morning Class"
    )
    assert success is True
    assert "marked successfully" in msg.lower()
    
    # 2. Re-attempt to mark duplicate in same session (should be rejected)
    success_dup, msg_dup = process_attendance_event(
        student_id="ST777",
        confidence=0.91,
        session="Morning Class"
    )
    assert success_dup is False
    assert "already marked" in msg_dup.lower()
    
    # 3. Mark attendance in a DIFFERENT session on the same day (should be allowed)
    success_diff_sess, msg_diff = process_attendance_event(
        student_id="ST777",
        confidence=0.85,
        session="Afternoon Class"
    )
    assert success_diff_sess is True
    
    # 4. Verify count of records is 2
    records = get_attendance_records()
    assert len(records) == 2
    assert records[0]["student_id"] == "ST777"

def test_unknown_face_attendance_rejection():
    # Attempt to mark attendance for "Unknown" identity
    success, msg = process_attendance_event(
        student_id="Unknown",
        confidence=0.35,
        session="Morning Class"
    )
    assert success is False
    assert "unknown face" in msg.lower()
    
    # Verify no records are written to attendance table
    records = get_attendance_records()
    assert len(records) == 0

def test_delete_attendance_record():
    from src.database import delete_attendance_record
    
    # 1. Create a record
    success, msg = process_attendance_event(
        student_id="ST777",
        confidence=0.95,
        session="Test Session"
    )
    assert success is True
    
    records = get_attendance_records()
    assert len(records) == 1
    record_id = records[0]["id"]
    
    # 2. Delete the record
    delete_success = delete_attendance_record(record_id)
    assert delete_success is True
    
    # 3. Verify it is gone
    records_after = get_attendance_records()
    assert len(records_after) == 0
