import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from datetime import datetime
import os
import sys

# Add base directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.analytics import get_dashboard_summary, get_weekly_attendance_trend, get_section_attendance
from src.database import get_attendance_records

def show():
    st.title("📊 Dashboard")
    st.markdown("### Deep Learning Powered Attendance Management")
    
    # Fetch summary stats from SQLite
    stats = get_dashboard_summary()
    
    # 1. KPI cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="Total Registered Students", value=stats["total_students"])
    with col2:
        st.metric(label="Present Today", value=stats["present_today"])
    with col3:
        st.metric(label="Absent Today", value=stats["absent_today"])
    with col4:
        st.metric(label="Attendance Rate", value=f"{stats['attendance_pct']}%")
        
    st.markdown("---")
    
    # 2. Charts Section
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.subheader("📈 Weekly Attendance Trend")
        trend_df = get_weekly_attendance_trend()
        if not trend_df.empty and trend_df["present_count"].sum() > 0:
            fig = px.line(
                trend_df, x="day_name", y="present_count", 
                labels={"day_name": "Day of Week", "present_count": "Present Count"},
                markers=True, title="Present Students (Last 7 Days)"
            )
            fig.update_layout(height=350, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No attendance recorded in the last 7 days. Mark attendance to view trend charts.")
            
    with col_chart2:
        st.subheader("🍰 Today's Attendance")
        if stats["total_students"] > 0:
            pie_data = pd.DataFrame({
                "Status": ["Present", "Absent"],
                "Count": [stats["present_today"], stats["absent_today"]]
            })
            fig = px.pie(
                pie_data, values="Count", names="Status", 
                color="Status", color_discrete_map={"Present": "#2ECC71", "Absent": "#E74C3C"},
                title="Overall Attendance Share"
            )
            fig.update_layout(height=350, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No students registered. Please go to the 'Register Student' page to start.")
            
    st.markdown("---")
    
    # 3. Section wise attendance and recent logs
    col_sec, col_logs = st.columns([1, 1])
    
    with col_sec:
        st.subheader("🏫 Section-Wise Breakdown")
        sec_df = get_section_attendance()
        if not sec_df.empty and sec_df["total_count"].sum() > 0:
            fig = px.bar(
                sec_df, x="section", y=["present_count", "absent_count"],
                labels={"value": "Students", "section": "Section", "variable": "Status"},
                color_discrete_map={"present_count": "#2ECC71", "absent_count": "#E74C3C"},
                title="Section Attendance Ratio (Today)",
                barmode="stack"
            )
            fig.update_layout(height=350, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No section data available.")
            
    with col_logs:
        st.subheader("⏱️ Recent Logs")
        recent_records = get_attendance_records()
        if recent_records:
            df_logs = pd.DataFrame(recent_records).head(5)
            # Display readable summary of logs
            for _, row in df_logs.iterrows():
                st.write(f"🔔 **{row['name']}** ({row['roll_number']}) was marked **{row['status']}** at {row['time']} on {row['date']} (Conf: {row['confidence']*100:.1f}%)")
        else:
            st.info("No attendance records found.")
