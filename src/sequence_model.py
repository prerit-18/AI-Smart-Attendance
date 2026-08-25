import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers
import sqlite3
import pandas as pd

# Add base directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import SEQUENCE_WINDOW_SIZE, SEQUENCE_MODEL_DIR
from src.database import get_db_connection

def get_student_attendance_sequence(student_id):
    """
    Constructs a chronological binary sequence of attendance for a student.
    1.0 = Present, 0.0 = Absent.
    Uses all distinct dates in the attendance table as the global timeline.
    """
    conn = get_db_connection()
    
    # Get all distinct attendance dates in chronological order
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT date FROM attendance ORDER BY date ASC")
    all_dates = [row[0] for row in cursor.fetchall()]
    
    if not all_dates:
        conn.close()
        return []
        
    # Get dates where this specific student was present
    cursor.execute("SELECT date FROM attendance WHERE student_id = ? AND status = 'Present'", (student_id,))
    present_dates = set([row[0] for row in cursor.fetchall()])
    conn.close()
    
    # Map dates to binary values (1.0 if present, 0.0 if absent)
    sequence = [1.0 if date in present_dates else 0.0 for date in all_dates]
    return sequence

def generate_synthetic_attendance_data(num_students=50, num_days=30):
    """
    Generates synthetic attendance records to train/evaluate the LSTM model (Course 5 demonstration).
    Creates students with varying attendance patterns (regular, irregular, weekly cycles).
    """
    np.random.seed(42)
    students = [f"ST{i:03d}" for i in range(1, num_students + 1)]
    
    # Generate dates (excluding weekends)
    dates = []
    current_date = pd.Timestamp.now() - pd.Timedelta(days=num_days)
    while len(dates) < num_days:
        if current_date.weekday() < 5:  # Monday to Friday
            dates.append(current_date.strftime("%Y-%m-%d"))
        current_date += pd.Timedelta(days=1)
        
    records = []
    for student_id in students:
        # Assign an attendance probability profile
        profile_type = np.random.choice(["regular", "average", "irregular"], p=[0.4, 0.4, 0.2])
        if profile_type == "regular":
            base_prob = 0.90
        elif profile_type == "average":
            base_prob = 0.75
        else:
            base_prob = 0.45
            
        for i, date_str in enumerate(dates):
            # Introduce some weekly pattern (e.g., lower attendance on Fridays)
            weekday = datetime_from_str = pd.Timestamp(date_str).weekday()
            prob = base_prob
            if weekday == 4:  # Friday
                prob -= 0.1
                
            # Random draw
            is_present = np.random.rand() < prob
            if is_present:
                records.append({
                    "student_id": student_id,
                    "date": date_str,
                    "time": "09:00:00",
                    "session": "Default",
                    "confidence": float(np.random.uniform(0.85, 0.99)),
                    "status": "Present"
                })
                
    return pd.DataFrame(records), dates, students

def prepare_lstm_data(sequences, window_size=SEQUENCE_WINDOW_SIZE):
    """
    Prepares inputs X and targets y for the LSTM network using sliding windows.
    X: shape (num_samples, window_size, 1)
    y: shape (num_samples, 1)
    """
    X, y = [], []
    for seq in sequences:
        if len(seq) <= window_size:
            continue
        for i in range(len(seq) - window_size):
            X.append(seq[i:i+window_size])
            y.append(seq[i+window_size])
            
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    
    if len(X) > 0:
        X = np.expand_dims(X, axis=-1)  # Reshape for LSTM: (N, W, 1)
        y = np.expand_dims(y, axis=-1)  # Reshape to (N, 1)
        
    return X, y

def build_lstm_model(window_size=SEQUENCE_WINDOW_SIZE):
    """
    Creates a simple LSTM sequence model using Keras.
    Demonstrates Sequence Models (Course 5) concepts.
    """
    model = tf.keras.Sequential([
        layers.Input(shape=(window_size, 1)),
        layers.LSTM(16, return_sequences=False, activation='tanh'),
        layers.Dense(8, activation='relu'),
        layers.Dense(1, activation='sigmoid')  # Outputs probability of presence
    ])
    
    # Compile with Adam optimizer (Course 2) and Binary Crossentropy
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

def train_sequence_model(X, y, epochs=15, batch_size=32):
    """
    Trains the LSTM model and saves it.
    Returns:
        history: training history dictionary
    """
    if len(X) == 0:
        return None
        
    model = build_lstm_model(window_size=X.shape[1])
    
    # Split train/val (Course 3 methodology)
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val) if len(X_val) > 0 else None,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0
    )
    
    # Save the trained model
    os.makedirs(SEQUENCE_MODEL_DIR, exist_ok=True)
    model.save(os.path.join(SEQUENCE_MODEL_DIR, "lstm_attendance_model.h5"))
    
    return history.history

def predict_next_attendance(student_id, window_size=SEQUENCE_WINDOW_SIZE):
    """
    Predicts the likelihood of a student being present on the next class day.
    """
    sequence = get_student_attendance_sequence(student_id)
    
    if len(sequence) < window_size:
        # Insufficient data, fallback to historical average rate
        if len(sequence) > 0:
            return float(np.mean(sequence)), "Based on historical average (insufficient sequence)"
        return 0.75, "Default prior probability (no records)"
        
    # Get the last W elements
    recent_seq = np.array(sequence[-window_size:], dtype=np.float32)
    recent_seq = np.reshape(recent_seq, (1, window_size, 1))
    
    model_path = os.path.join(SEQUENCE_MODEL_DIR, "lstm_attendance_model.h5")
    if os.path.exists(model_path):
        try:
            model = tf.keras.models.load_model(model_path)
            pred = model.predict(recent_seq, verbose=0)[0][0]
            return float(pred), "Predicted by LSTM sequence model"
        except Exception as e:
            print(f"Error loading LSTM model: {e}")
            
    # Fallback if model not trained/loaded
    return float(np.mean(recent_seq)), "Based on rolling average (model not loaded)"
