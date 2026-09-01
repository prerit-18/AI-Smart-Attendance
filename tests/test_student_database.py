import pytest
import os
import sys

# Add base directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config.config as config
from src.database import (
    init_db, register_student, get_student,
    update_student, delete_student_embeddings
)
from src.pages.student_database import (
    get_student_samples, get_next_sample_filename
)

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
    # Register test students
    register_student(
        student_id="ST101",
        name="Original Name",
        roll_number="101001",
        section="CSE-A",
        department="Engineering",
        email="orig@univ.edu"
    )
    register_student(
        student_id="ST102",
        name="Second Student",
        roll_number="101002",
        section="CSE-B",
        department="Engineering",
        email="second@univ.edu"
    )
    yield
    if os.path.exists(test_db_path):
        try:
            os.remove(test_db_path)
        except Exception:
            pass

def test_update_student_details():
    # Update student details
    success, msg = update_student(
        student_id="ST101",
        name="Updated Name",
        roll_number="101099",
        section="CSE-C",
        department="Computer Science",
        email="updated@univ.edu"
    )
    assert success is True
    assert "updated successfully" in msg.lower()

    # Retrieve and verify updated fields
    student = get_student("ST101")
    assert student is not None
    assert student["name"] == "Updated Name"
    assert student["roll_number"] == "101099"
    assert student["section"] == "CSE-C"
    assert student["department"] == "Computer Science"
    assert student["email"] == "updated@univ.edu"

def test_update_student_duplicate_roll_number():
    # Attempt to update ST101's roll number to ST102's roll number (101002)
    success, msg = update_student(
        student_id="ST101",
        name="Updated Name",
        roll_number="101002",
        section="CSE-A",
        department="Engineering",
        email="updated@univ.edu"
    )
    assert success is False
    assert "already assigned" in msg.lower()

    # Verify original details remain intact
    student = get_student("ST101")
    assert student["roll_number"] == "101001"

def test_update_nonexistent_student():
    success, msg = update_student(
        student_id="ST9999",
        name="Ghost",
        roll_number="999999",
        section="NONE",
        department="NONE",
        email="ghost@univ.edu"
    )
    assert success is False
    assert "not found" in msg.lower()

def test_get_next_sample_filename(tmp_path):
    test_sid = "ST_SAMPLE_TEST"
    test_dir = os.path.join(config.STUDENTS_DIR, test_sid)
    os.makedirs(test_dir, exist_ok=True)
    try:
        # With 0 samples
        assert get_next_sample_filename(test_sid) == "sample_1.png"

        # Create sample_1.png and sample_2.png
        with open(os.path.join(test_dir, "sample_1.png"), "w") as f:
            f.write("test")
        with open(os.path.join(test_dir, "sample_2.png"), "w") as f:
            f.write("test")

        assert len(get_student_samples(test_sid)) == 2
        assert get_next_sample_filename(test_sid) == "sample_3.png"
    finally:
        import shutil
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
