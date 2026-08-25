import os

# Base Directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Database Config
DB_DIR = os.path.join(BASE_DIR, "database")
os.makedirs(DB_DIR, exist_ok=True)
DATABASE_PATH = os.path.join(DB_DIR, "attendance.db")

# Data Config
DATA_DIR = os.path.join(BASE_DIR, "data")
STUDENTS_DIR = os.path.join(DATA_DIR, "students")
os.makedirs(STUDENTS_DIR, exist_ok=True)

# Models Config
MODELS_DIR = os.path.join(BASE_DIR, "models")
FACE_MODEL_DIR = os.path.join(MODELS_DIR, "face_model")
EMBEDDINGS_DIR = os.path.join(MODELS_DIR, "embeddings")
SEQUENCE_MODEL_DIR = os.path.join(MODELS_DIR, "sequence_model")
os.makedirs(FACE_MODEL_DIR, exist_ok=True)
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
os.makedirs(SEQUENCE_MODEL_DIR, exist_ok=True)

# Results Config
RESULTS_DIR = os.path.join(BASE_DIR, "results")
TRAINING_HISTORY_DIR = os.path.join(RESULTS_DIR, "training_history")
CONFUSION_MATRIX_DIR = os.path.join(RESULTS_DIR, "confusion_matrix")
METRICS_DIR = os.path.join(RESULTS_DIR, "metrics")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
os.makedirs(TRAINING_HISTORY_DIR, exist_ok=True)
os.makedirs(CONFUSION_MATRIX_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# Model Training and Image Settings
IMAGE_SIZE = (160, 160)  # Standard input size for MobileNetV2/FaceNet
MIN_FACE_QUALITY = 50     # Min face box dimension in pixels to filter blurry/far detections

# Recognition Settings
RECOGNITION_THRESHOLD = 0.6  # Cosine similarity threshold for face recognition (higher = more strict)
CAMERA_INDEX = 0             # Default webcam index

# Sequence Model Config (Course 5)
SEQUENCE_WINDOW_SIZE = 5     # Time steps window to predict the next attendance state

# Demo/Sample Settings
DEMO_MODE = False            # Can be toggled in Streamlit sidebar
