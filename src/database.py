import sqlite3
import json
import os
import numpy as np
from datetime import datetime
import sys

# Add base directory to path so config can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config.config as config

def get_db_connection():
    """Establishes and returns a connection to the SQLite database."""
    conn = sqlite3.connect(config.DATABASE_PATH)
    conn.row_factory = sqlite3.Row  # Access columns by name
    return conn

def init_db():
    """Initializes the database schema and creates necessary tables and indexes."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # 1. Students Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS students (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_id TEXT UNIQUE NOT NULL,
        name TEXT NOT NULL,
        roll_number TEXT UNIQUE NOT NULL,
        section TEXT NOT NULL,
        department TEXT NOT NULL,
        email TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    
    # 2. Face Embeddings Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS face_embeddings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_id TEXT NOT NULL,
        embedding TEXT NOT NULL,  -- JSON string of list
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (student_id) REFERENCES students(student_id) ON DELETE CASCADE
    )
    """)
    
    # 3. Attendance Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS attendance (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_id TEXT NOT NULL,
        date TEXT NOT NULL,       -- YYYY-MM-DD
        time TEXT NOT NULL,       -- HH:MM:SS
        session TEXT NOT NULL,    -- Default/Morning/Afternoon
        confidence REAL NOT NULL,
        status TEXT NOT NULL,     -- Present/Absent/Late
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (student_id) REFERENCES students(student_id) ON DELETE CASCADE
    )
    """)
    
    # 4. Recognition Logs Table (Audit trail)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS recognition_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        student_id TEXT,
        confidence REAL,
        recognition_status TEXT NOT NULL  -- Success / Failure (Unknown)
    )
    """)
    
    # Create Indexes for performance optimization
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_students_student_id ON students(student_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_embeddings_student_id ON face_embeddings(student_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_attendance_composite ON attendance(student_id, date, session)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_attendance_date ON attendance(date)")
    
    conn.commit()
    conn.close()

def register_student(student_id, name, roll_number, section, department, email):
    """Registers a new student. Returns True if successful, False if duplicate student_id/roll_number."""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
        INSERT INTO students (student_id, name, roll_number, section, department, email)
        VALUES (?, ?, ?, ?, ?, ?)
        """, (student_id, name, roll_number, section, department, email))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def delete_student(student_id):
    """Deletes a student and their face embeddings, attendance records from SQLite."""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        # Enable Foreign Keys to trigger Cascade deletes
        cursor.execute("PRAGMA foreign_keys = ON")
        cursor.execute("DELETE FROM students WHERE student_id = ?", (student_id,))
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()

def get_student(student_id):
    """Retrieves student details by ID."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM students WHERE student_id = ?", (student_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None

def get_all_students():
    """Retrieves all registered students."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM students ORDER BY created_at DESC")
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def save_embedding(student_id, embedding_vector):
    """Saves a 1D numpy face embedding array as a serialized JSON list."""
    conn = get_db_connection()
    cursor = conn.cursor()
    # Serialize embedding list to JSON
    embedding_str = json.dumps(embedding_vector.tolist())
    cursor.execute("""
    INSERT INTO face_embeddings (student_id, embedding)
    VALUES (?, ?)
    """, (student_id, embedding_str))
    conn.commit()
    conn.close()

def get_student_embeddings(student_id):
    """Retrieves all embeddings for a specific student."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT embedding FROM face_embeddings WHERE student_id = ?", (student_id,))
    rows = cursor.fetchall()
    conn.close()
    
    embeddings = []
    for row in rows:
        embeddings.append(np.array(json.loads(row['embedding']), dtype=np.float32))
    return embeddings

def get_all_embeddings():
    """Retrieves all stored face embeddings and maps them to student IDs."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT student_id, embedding FROM face_embeddings")
    rows = cursor.fetchall()
    conn.close()
    
    embeddings_dict = {}
    for row in rows:
        sid = row['student_id']
        emb = np.array(json.loads(row['embedding']), dtype=np.float32)
        if sid not in embeddings_dict:
            embeddings_dict[sid] = []
        embeddings_dict[sid].append(emb)
    return embeddings_dict

def is_attendance_marked(student_id, date_str, session_str):
    """Checks if attendance is already marked for a student on a specific date and session."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
    SELECT 1 FROM attendance 
    WHERE student_id = ? AND date = ? AND session = ?
    """, (student_id, date_str, session_str))
    marked = cursor.fetchone() is not None
    conn.close()
    return marked

def mark_attendance(student_id, date_str, time_str, session_str, confidence, status="Present"):
    """Marks attendance for a student if not already marked."""
    if is_attendance_marked(student_id, date_str, session_str):
        return False, "Attendance already marked."
    
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
        INSERT INTO attendance (student_id, date, time, session, confidence, status)
        VALUES (?, ?, ?, ?, ?, ?)
        """, (student_id, date_str, time_str, session_str, confidence, status))
        conn.commit()
        return True, "Attendance marked successfully."
    except Exception as e:
        return False, str(e)
    finally:
        conn.close()

def log_recognition(student_id, confidence, recognition_status):
    """Logs a face recognition event to the database for auditing and analytics."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
    INSERT INTO recognition_logs (student_id, confidence, recognition_status)
    VALUES (?, ?, ?)
    """, (student_id, confidence, recognition_status))
    conn.commit()
    conn.close()

def get_attendance_records(date_filter=None, student_filter=None, section_filter=None, status_filter=None):
    """Retrieves filtered attendance records with student name and roll number."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    query = """
    SELECT a.id, a.student_id, s.name, s.roll_number, s.section, s.department, a.date, a.time, a.session, a.confidence, a.status 
    FROM attendance a
    JOIN students s ON a.student_id = s.student_id
    WHERE 1=1
    """
    params = []
    
    if date_filter:
        query += " AND a.date = ?"
        params.append(date_filter)
    if student_filter:
        query += " AND (a.student_id = ? OR s.name LIKE ?)"
        params.extend([student_filter, f"%{student_filter}%"])
    if section_filter:
        query += " AND s.section = ?"
        params.append(section_filter)
    if status_filter:
        query += " AND a.status = ?"
        params.append(status_filter)
        
    query += " ORDER BY a.date DESC, a.time DESC"
    
    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def delete_attendance_record(record_id):
    """Deletes a specific attendance log entry by its unique ID."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM attendance WHERE id = ?", (record_id,))
    conn.commit()
    conn.close()
    return True
