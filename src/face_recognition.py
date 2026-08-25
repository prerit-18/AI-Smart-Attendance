import os
import sys
import numpy as np

# Add base directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import RECOGNITION_THRESHOLD
from src.database import get_all_embeddings
from src.embeddings import compute_similarity

# Global dictionary to cache student embeddings: {student_id: [embedding_vector, ...]}
_KNOWN_FACE_EMBEDDINGS = {}

def load_known_face_embeddings(force_reload=False):
    """
    Loads all student embeddings from the SQLite database and caches them.
    """
    global _KNOWN_FACE_EMBEDDINGS
    if not _KNOWN_FACE_EMBEDDINGS or force_reload:
        _KNOWN_FACE_EMBEDDINGS = get_all_embeddings()
    return _KNOWN_FACE_EMBEDDINGS

def recognize_face(embedding, threshold=None):
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
        
    known_embeddings = load_known_face_embeddings()
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
