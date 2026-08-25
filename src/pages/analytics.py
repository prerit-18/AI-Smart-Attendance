import streamlit as st
import pandas as pd
import plotly.express as px
import os
import sys

# Add base directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.analytics import get_student_ranking, get_section_attendance, get_weekly_attendance_trend
from src.sequence_model import (
    predict_next_attendance, train_sequence_model, 
    prepare_lstm_data, generate_synthetic_attendance_data
)
from src.database import get_all_students

def show():
    st.title("📈 Attendance Analytics & Sequence Models")
    st.markdown("### Deep Learning Attendance Patterns and Predictive Modeling")
    
    tab_stats, tab_lstm = st.tabs(["📊 Performance Statistics", "🧠 Course 5: LSTM Sequence Predictions"])
    
    students = get_all_students()
    
    with tab_stats:
        if not students:
            st.warning("No student records available. Register students and mark attendance to see analytics.")
            return
            
        st.subheader("Student Attendance Rankings")
        col_high, col_low = st.columns(2)
        
        with col_high:
            st.markdown("🏆 **Most Regular Students (Top 5)**")
            df_high = get_student_ranking(limit=5, ascending=False)
            if not df_high.empty:
                st.dataframe(df_high[["name", "roll_number", "section", "days_present", "attendance_rate"]], use_container_width=True)
            else:
                st.write("No attendance recorded yet.")
                
        with col_low:
            st.markdown("⚠️ **Students with Low Attendance (Alert list)**")
            df_low = get_student_ranking(limit=5, ascending=True)
            if not df_low.empty:
                # Filter out those with 0 presence if no records at all
                st.dataframe(df_low[["name", "roll_number", "section", "days_present", "attendance_rate"]], use_container_width=True)
            else:
                st.write("No attendance recorded yet.")
                
        st.markdown("---")
        
        # Section comparisons
        st.subheader("Class Section Comparisons")
        sec_df = get_section_attendance()
        if not sec_df.empty:
            # Calculate actual rate per section
            sec_df["rate"] = (sec_df["present_count"] / sec_df["total_count"] * 100).round(1)
            fig_rate = px.bar(
                sec_df, x="section", y="rate",
                labels={"rate": "Attendance Rate (%)", "section": "Section"},
                title="Section Attendance Rates (%) Today",
                color="rate", color_continuous_scale="RdYlGn"
            )
            st.plotly_chart(fig_rate, use_container_width=True)
        else:
            st.info("No section statistics available.")
            
    with tab_lstm:
        st.subheader("Temporal Sequence Modeling (LSTM)")
        st.markdown(
            """
            Attendance is time-series data (sequences of presence/absence). 
            Using concepts from **Course 5 (Sequence Models)**, we represent each student's history as a binary vector (e.g. `[1, 1, 0, 1]`) 
            and feed it into an LSTM network to predict the probability of attending the next class session.
            """
        )
        
        # 1. Trigger LSTM Training
        st.markdown("### 🧪 Model Training & Hyperparameter Setup")
        
        col_epochs, col_btn = st.columns([2, 1])
        with col_epochs:
            epochs = st.slider("LSTM Training Epochs", 5, 50, 15)
        with col_btn:
            st.write("") # Spacer
            st.write("") # Spacer
            train_click = st.button("Run LSTM Training Loop")
            
        if train_click:
            # Check if we have real attendance data
            # To train an LSTM we need sequences, so we generate synthetic historical data to pre-train
            # if the active database is empty, demonstrating standard Course 5 logic.
            st.write("🔄 Preparing sequences from database...")
            
            # Retrieve chronological sequences for registered students
            sequences = []
            for s in students:
                seq = get_student_attendance_sequence_for_lstm(s['student_id'])
                if len(seq) > 5:
                    sequences.append(seq)
                    
            is_synthetic_used = False
            if len(sequences) < 5 or max([len(seq) for seq in sequences]) <= 5:
                st.info("ℹ️ Insufficient active attendance history for LSTM training (sequences need length > 5). Falling back to synthetic history dataset.")
                df_synth, dates_synth, students_synth = generate_synthetic_attendance_data(num_students=30, num_days=40)
                
                # Reconstruct sequences from synthetic DataFrame
                synth_sequences = []
                for sid in students_synth:
                    present_dates = set(df_synth[(df_synth["student_id"] == sid) & (df_synth["status"] == "Present")]["date"])
                    seq = [1.0 if d in present_dates else 0.0 for d in dates_synth]
                    synth_sequences.append(seq)
                sequences = synth_sequences
                is_synthetic_used = True
                
            X, y = prepare_lstm_data(sequences)
            
            if len(X) == 0:
                st.error("❌ Failed to construct training windows.")
            else:
                st.write(f"📊 Extracted **{len(X)} training samples** (Sliding Window Size = 5).")
                st.write("⚙️ Compiling LSTM architecture: `Input -> LSTM(16) -> Dense(8) -> Dense(1, Sigmoid)`...")
                
                # Run fit
                with st.spinner("Training LSTM (Adam optimizer)..."):
                    history = train_sequence_model(X, y, epochs=epochs, batch_size=16)
                    
                if history:
                    st.success("🎉 LSTM Sequence Model trained and saved to disk!")
                    
                    # Plot LSTM learning curve
                    fig_loss = px.line(
                        y=history['loss'], 
                        labels={"x": "Epochs", "y": "Binary Crossentropy Loss"},
                        title="LSTM Sequence Training Loss Curve"
                    )
                    st.plotly_chart(fig_loss, use_container_width=True)
                else:
                    st.error("Training failed.")
                    
        st.markdown("---")
        
        # 2. Individual Student Predictions
        st.markdown("### 🔮 Next-Session Attendance Predictor")
        if not students:
            st.info("Please register students first.")
            return
            
        selected_student = st.selectbox(
            "Select Student to Analyze", 
            [f"{s['name']} ({s['student_id']})" for s in students]
        )
        
        sid = selected_student.split("(")[-1].replace(")", "").strip()
        
        # Display sequence
        seq = get_student_attendance_sequence_for_lstm(sid)
        st.write(f"**Chronological Attendance History (Last 10 classes)**:")
        
        if not seq:
            st.write("`[No history recorded yet]`")
        else:
            display_seq = [f"🟢 Present" if val == 1.0 else "🔴 Absent" for val in seq[-10:]]
            st.write(f"`{' -> '.join(display_seq)}`")
            
        # Run prediction
        prob, method = predict_next_attendance(sid)
        
        col_metric, col_desc = st.columns([1, 2])
        with col_metric:
            st.metric(
                label="Predicted Presence Likelihood", 
                value=f"{prob*100:.1f}%", 
                delta=f"{'High chance' if prob >= 0.7 else 'At risk'}",
                delta_color="normal" if prob >= 0.7 else "inverse"
            )
        with col_desc:
            st.write(f"**Calculation Basis**: {method}")
            st.markdown(
                f"""
                A score of **{prob*100:.1f}%** indicates the student has a {'strong' if prob >= 0.7 else 'weak'} likelihood of attending the next class, 
                based on temporal scheduling habits, weekend lapses, and cyclical attendance patterns captured by our models.
                """
            )

def get_student_attendance_sequence_for_lstm(student_id):
    """Local helper to get binary sequence."""
    from src.sequence_model import get_student_attendance_sequence
    return get_student_attendance_sequence(student_id)
