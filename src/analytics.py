import os
import sys
import pandas as pd
import sqlite3
from datetime import datetime, timedelta

# Add base directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.database import get_db_connection

def get_dashboard_summary():
    """
    Computes high-level KPI metrics for the home dashboard.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    today_str = datetime.now().strftime("%Y-%m-%d")
    
    # 1. Total Registered Students
    cursor.execute("SELECT COUNT(*) FROM students")
    total_students = cursor.fetchone()[0]
    
    # 2. Present Today
    cursor.execute("SELECT COUNT(DISTINCT student_id) FROM attendance WHERE date = ?", (today_str,))
    present_today = cursor.fetchone()[0]
    
    # 3. Absent Today
    absent_today = max(0, total_students - present_today)
    
    # 4. Attendance Percentage Today
    attendance_pct = (present_today / total_students * 100) if total_students > 0 else 0.0
    
    # 5. Total Attendance Records
    cursor.execute("SELECT COUNT(*) FROM attendance")
    total_records = cursor.fetchone()[0]
    
    conn.close()
    
    return {
        "total_students": total_students,
        "present_today": present_today,
        "absent_today": absent_today,
        "attendance_pct": round(attendance_pct, 1),
        "total_records": total_records
    }

def get_weekly_attendance_trend():
    """
    Fetches attendance counts for the last 7 calendar days.
    """
    conn = get_db_connection()
    
    # Generate list of last 7 dates
    dates = [(datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(6, -1, -1)]
    
    query = """
    SELECT date, COUNT(DISTINCT student_id) as present_count 
    FROM attendance 
    WHERE date >= ? 
    GROUP BY date
    """
    min_date = dates[0]
    df = pd.read_sql_query(query, conn, params=[min_date])
    conn.close()
    
    # Create complete dataframe including dates with 0 attendance
    trend_df = pd.DataFrame({"date": dates})
    trend_df = trend_df.merge(df, on="date", how="left").fillna(0)
    trend_df["present_count"] = trend_df["present_count"].astype(int)
    
    # Convert date strings to friendly weekday names
    trend_df["day_name"] = trend_df["date"].apply(
        lambda x: datetime.strptime(x, "%Y-%m-%d").strftime("%a")
    )
    
    return trend_df

def get_section_attendance():
    """
    Computes attendance counts broken down by section/class.
    """
    conn = get_db_connection()
    query = """
    SELECT s.section, COUNT(DISTINCT a.student_id) as present_count
    FROM attendance a
    JOIN students s ON a.student_id = s.student_id
    WHERE a.date = ?
    GROUP BY s.section
    """
    today_str = datetime.now().strftime("%Y-%m-%d")
    df = pd.read_sql_query(query, conn, params=[today_str])
    
    # Get total registered students per section to show ratios
    query_totals = "SELECT section, COUNT(*) as total_count FROM students GROUP BY section"
    df_totals = pd.read_sql_query(query_totals, conn)
    conn.close()
    
    if df.empty:
        df = pd.DataFrame(columns=["section", "present_count"])
        
    merged = df_totals.merge(df, on="section", how="left").fillna(0)
    merged["present_count"] = merged["present_count"].astype(int)
    merged["absent_count"] = merged["total_count"] - merged["present_count"]
    return merged

def get_student_ranking(limit=10, ascending=False):
    """
    Ranks students based on their cumulative attendance count.
    Useful for finding 'most regular' or 'low attendance' students.
    """
    conn = get_db_connection()
    
    # Get total active attendance days
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(DISTINCT date) FROM attendance")
    total_days = cursor.fetchone()[0]
    total_days = max(1, total_days)  # Avoid division by zero
    
    query = """
    SELECT s.student_id, s.name, s.roll_number, s.section, COUNT(a.id) as days_present
    FROM students s
    LEFT JOIN attendance a ON s.student_id = a.student_id
    GROUP BY s.student_id
    ORDER BY days_present DESC, s.name ASC
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    df["attendance_rate"] = (df["days_present"] / total_days * 100).round(1)
    
    if ascending:
        df = df.sort_values(by="attendance_rate", ascending=True)
    else:
        df = df.sort_values(by="attendance_rate", ascending=False)
        
    return df.head(limit)
