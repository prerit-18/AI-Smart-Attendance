import os
import sys
import numpy as np

# Add base directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import RECOGNITION_THRESHOLD
from src.database import get_all_embeddings
from src.embeddings import compute_similarity

# Global dictionary to cache student embeddings: {(model_type, expected_dim): {student_id: [embedding_vector, ...]}}
_KNOWN_FACE_EMBEDDINGS = {}

def load_known_face_embeddings(model_type=None, expected_dim=None, force_reload=False):
    """
    Loads student embeddings from SQLite, filtered by model and/or dimension, and caches them.
    """
    global _KNOWN_FACE_EMBEDDINGS
    cache_key = (model_type, expected_dim)
    if force_reload or cache_key not in _KNOWN_FACE_EMBEDDINGS:
        embs = get_all_embeddings(model_name=model_type, expected_dim=expected_dim)
        # If no embeddings found for specific model_name, fallback to any embedding with matching dimension
        if not embs and expected_dim is not None:
            embs = get_all_embeddings(model_name=None, expected_dim=expected_dim)
        _KNOWN_FACE_EMBEDDINGS[cache_key] = embs
    return _KNOWN_FACE_EMBEDDINGS[cache_key]

def recognize_face(embedding, threshold=None, model_type=None):
    """
    Compares the input face embedding against all registered student embeddings.
    Returns:
        student_id: str or "Unknown"
        confidence: float (0.0 to 1.0) representation of similarity
        is_known: bool
    """
    if embedding is None:
        return "Unknown", 0.0, False
        
    if threshold is None:
        threshold = RECOGNITION_THRESHOLD
        
    expected_dim = len(embedding)
    known_embeddings = load_known_face_embeddings(model_type=model_type, expected_dim=expected_dim)
    if not known_embeddings:
        return "Unknown", 0.0, False
        
    best_student_id = "Unknown"
    best_similarity = -1.0
    
    # Compare with each student's stored embeddings
    for student_id, student_embs in known_embeddings.items():
        for known_emb in student_embs:
            sim = compute_similarity(embedding, known_emb)
            if sim > best_similarity:
                best_similarity = sim
                best_student_id = student_id
                
    # If best match exceeds the recognition threshold, classify as that student
    if best_similarity >= threshold:
        return best_student_id, best_similarity, True
    else:
        return "Unknown", best_similarity, False
