import pytest
import os
import sqlite3
import sys

# Add base directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config.config as config

# Import database functions
from src.database import (
    init_db, register_student, delete_student, 
    get_student, get_all_students, save_embedding, get_student_embeddings
)

@pytest.fixture(autouse=True)
def setup_test_db(request):
    """Initializes test-specific database before each test case runs and cleans it up after."""
    test_name = request.node.name
    test_db_path = os.path.join(config.DB_DIR, f"test_db_{test_name}.db")
    config.DATABASE_PATH = test_db_path
    
    if os.path.exists(test_db_path):
        try:
            os.remove(test_db_path)
        except Exception:
            pass
            
    init_db()
    yield
    
    # Cleanup after test
    if os.path.exists(test_db_path):
        try:
            os.remove(test_db_path)
        except Exception:
            pass

def test_student_registration_and_retrieval():
    # 1. Register a student
    success = register_student(
        student_id="ST999",
        name="Test Student",
        roll_number="999000",
        section="TEST-A",
        department="QA",
        email="test@qa.com"
    )
    assert success is True
    
    # 2. Retrieve student details
    student = get_student("ST999")
    assert student is not None
    assert student["name"] == "Test Student"
    assert student["roll_number"] == "999000"
    assert student["section"] == "TEST-A"
    
    # 3. Retrieve all students
    all_students = get_all_students()
    assert len(all_students) == 1
    assert all_students[0]["student_id"] == "ST999"

def test_duplicate_student_prevention():
    # Register first student
    success1 = register_student("ST999", "Test 1", "999001", "A", "QA", "t1@qa.com")
    assert success1 is True
    
    # Register duplicate Student ID
    success2 = register_student("ST999", "Test 2", "999002", "B", "QA", "t2@qa.com")
    assert success2 is False
    
    # Register duplicate Roll Number
    success3 = register_student("ST998", "Test 3", "999001", "C", "QA", "t3@qa.com")
    assert success3 is False

def test_student_deletion_cascades():
    # Register student
    register_student("ST999", "Test", "999003", "A", "QA", "t@qa.com")
    
    # Save a dummy embedding
    import numpy as np
    dummy_emb = np.random.randn(128)
    save_embedding("ST999", dummy_emb)
    
    # Verify embedding is written
    embs = get_student_embeddings("ST999")
    assert len(embs) == 1
    
    # Delete student
    deleted = delete_student("ST999")
    assert deleted is True
    
    # Verify student is gone
    assert get_student("ST999") is None
    
    # Verify embeddings are deleted (cascade constraint)
    # Note: In SQLite in-memory, to trigger cascade deletes, foreign_keys pragma must be enabled
    # The delete_student function enables PRAGMA foreign_keys = ON, so let's verify cascade works
    embs_after = get_student_embeddings("ST999")
    assert len(embs_after) == 0
