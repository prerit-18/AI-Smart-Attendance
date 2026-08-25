import pytest
import os
import sys
import numpy as np

# Add base directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config.config as config
from src.database import init_db, register_student, save_embedding
from src.embeddings import compute_similarity
from src.face_recognition import recognize_face, load_known_face_embeddings

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
    # Register ST111
    register_student("ST111", "Recognize Target", "111000", "TEST-C", "QA", "target@qa.com")
    
    # Save a mock normalized 3-dimensional vector (padded with zeros to dimension 128)
    emb_vector = np.zeros(128, dtype=np.float32)
    emb_vector[0] = 1.0  # Unit vector on axis 0
    save_embedding("ST111", emb_vector)
    
    # Force reload cache to include new embedding
    load_known_face_embeddings(force_reload=True)
    yield
    if os.path.exists(test_db_path):
        try:
            os.remove(test_db_path)
        except Exception:
            pass

def test_cosine_similarity_math():
    # Cosine similarity of identical vectors is 1.0
    v1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    v2 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    assert abs(compute_similarity(v1, v2) - 1.0) < 1e-6
    
    # Cosine similarity of orthogonal vectors is 0.0
    v3 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    assert abs(compute_similarity(v1, v3) - 0.0) < 1e-6

def test_face_recognition_exact_match():
    # Query with exact matching vector (ST111 signature)
    query_emb = np.zeros(128, dtype=np.float32)
    query_emb[0] = 1.0
    
    student_id, confidence, is_known = recognize_face(query_emb, threshold=0.6)
    
    assert is_known is True
    assert student_id == "ST111"
    assert abs(confidence - 1.0) < 1e-5

def test_face_recognition_mismatch_under_threshold():
    # Query with orthogonal vector (similarity = 0.0, below threshold)
    query_emb = np.zeros(128, dtype=np.float32)
    query_emb[1] = 1.0  # Unit vector on axis 1
    
    student_id, confidence, is_known = recognize_face(query_emb, threshold=0.6)
    
    assert is_known is False
    assert student_id == "Unknown"
    assert abs(confidence - 0.0) < 1e-5
