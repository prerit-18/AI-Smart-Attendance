import matplotlib.pyplot as plt
import os
import sys

# Add base directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def draw_system_architecture():
    """Draws and saves the system architecture flowchart."""
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.axis('off')
    
    # Define box style
    box_blue = dict(boxstyle="round,pad=0.5", fc="#D5F5E3", ec="#27AE60", lw=2)
    box_green = dict(boxstyle="round,pad=0.5", fc="#D6EAF8", ec="#2980B9", lw=2)
    box_orange = dict(boxstyle="round,pad=0.5", fc="#FCF3CF", ec="#F39C12", lw=2)
    box_red = dict(boxstyle="round,pad=0.5", fc="#FADBD8", ec="#C0392B", lw=2)
    
    # 1. Main Recognition Pipeline Nodes
    nodes = {
        "webcam": (2, 9, "Webcam / Camera Feed\n(BGR Video Input)", box_blue),
        "detector": (2, 7.8, "Face Detection\n(OpenCV Haar Cascade)", box_blue),
        "preprocess": (2, 6.6, "Face Preprocessing\n(Resize to 160x160 & Scale [-1,1])", box_blue),
        "extractor": (2, 5.4, "CNN Feature Extractor\n(Pretrained MobileNetV2 / Custom CNN)", box_green),
        "embedding": (2, 4.2, "L2-Normalized Embedding\n(128-Dimensional Vector)", box_green),
        "matcher": (2, 3.0, "Cosine Similarity Matcher\n(Compare with Registered Embeddings)", box_green),
        "threshold": (2, 1.8, "Confidence Threshold Check\n(sim >= threshold)", box_orange),
        "db": (6, 1.8, "SQLite Database\n(students, attendance, logs)", box_red),
        "dashboard": (9, 1.8, "Streamlit Dashboard\n(KPI Cards, Charts, CSV Export)", box_red),
    }
    
    # 2. Sequence Analytics Pipeline Nodes
    seq_nodes = {
        "history": (6, 5.4, "Attendance History\n(Present/Absent Timelines)", box_orange),
        "lstm_prep": (6, 4.2, "Temporal Windowing\n(Sliding window W=5)", box_orange),
        "lstm": (9, 4.2, "LSTM Network (Keras)\n(Recurrent Predictor)", box_green),
        "prediction": (9, 3.0, "Future Presence Probability\n(Attendance Forecasting)", box_orange)
    }
    
    # Draw all main nodes
    for name, (x, y, text, box) in {**nodes, **seq_nodes}.items():
        ax.text(x, y, text, ha="center", va="center", bbox=box, fontsize=10)
        
    # Draw arrows for main pipeline
    arrow_props = dict(arrowstyle="->", color="#2C3E50", lw=1.5, mutation_scale=15)
    
    ax.annotate("", xy=(2, 8.2), xytext=(2, 8.7), arrowprops=arrow_props)
    ax.annotate("", xy=(2, 7.0), xytext=(2, 7.4), arrowprops=arrow_props)
    ax.annotate("", xy=(2, 5.8), xytext=(2, 6.2), arrowprops=arrow_props)
    ax.annotate("", xy=(2, 4.6), xytext=(2, 5.0), arrowprops=arrow_props)
    ax.annotate("", xy=(2, 3.4), xytext=(2, 3.8), arrowprops=arrow_props)
    ax.annotate("", xy=(2, 2.2), xytext=(2, 2.6), arrowprops=arrow_props)
    
    # Threshold to Database
    ax.annotate("", xy=(5.0, 1.8), xytext=(3.5, 1.8), arrowprops=arrow_props)
    # Database to Dashboard
    ax.annotate("", xy=(7.9, 1.8), xytext=(7.1, 1.8), arrowprops=arrow_props)
    
    # Database feeds History
    ax.annotate("", xy=(6, 5.0), xytext=(6, 2.2), arrowprops=arrow_props)
    # History to LSTM Prep
    ax.annotate("", xy=(6, 4.6), xytext=(6, 5.0), arrowprops=arrow_props)
    # LSTM Prep to LSTM Model
    ax.annotate("", xy=(8.1, 4.2), xytext=(6.9, 4.2), arrowprops=arrow_props)
    # LSTM Model to Predictions
    ax.annotate("", xy=(9, 3.4), xytext=(9, 3.8), arrowprops=arrow_props)
    # Predictions to Dashboard
    ax.annotate("", xy=(9, 2.2), xytext=(9, 2.6), arrowprops=arrow_props)
    
    # Title
    ax.text(5.5, 9.7, "AI SMART ATTENDANCE - SYSTEM ARCHITECTURE", ha="center", va="center", 
            fontsize=14, fontweight="bold", color="#2C3E50")
            
    plt.tight_layout()
    os.makedirs(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs"), exist_ok=True)
    save_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "system_architecture.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved System Architecture Diagram to {save_path}")

def draw_methodology():
    """Draws and saves the methodology flow diagram."""
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axis('off')
    
    box_style = dict(boxstyle="square,pad=0.6", fc="#EBEDEF", ec="#7F8C8D", lw=1.5)
    
    steps = [
        "1. Data Collection (Webcam Frame Captures)",
        "2. Data Preprocessing (RGB Rescale & Noise Normalization)",
        "3. Face Detection (Haar Cascades Region Proposal)",
        "4. Feature Extraction (Deep CNN Backbone Embedding)",
        "5. Model Development (Regularization & Transfer Selection)",
        "6. Model Training (Adam Optimizer Loss Minimization)",
        "7. Model Evaluation (Test Validation & Error Profiling)",
        "8. Face Recognition (Cosine Similarity Comparison)",
        "9. Attendance Recording (Duplicate Checks & Database Insert)",
        "10. Analytics (Forecasting & Streamlit Visual Metrics)"
    ]
    
    # Draw vertical steps
    y_pos = 9.0
    arrow_props = dict(arrowstyle="->", color="#7F8C8D", lw=1.5)
    
    for i, step in enumerate(steps):
        ax.text(5.0, y_pos, step, ha="center", va="center", bbox=box_style, fontsize=11, fontweight="semibold")
        if i < len(steps) - 1:
            ax.annotate("", xy=(5.0, y_pos - 0.55), xytext=(5.0, y_pos - 0.35), arrowprops=arrow_props)
        y_pos -= 0.85
        
    ax.text(5.0, 9.7, "AI SMART ATTENDANCE - METHODOLOGY STAGES", ha="center", va="center", 
            fontsize=13, fontweight="bold", color="#2C3E50")
            
    plt.tight_layout()
    save_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "methodology.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved Methodology Diagram to {save_path}")

if __name__ == "__main__":
    draw_system_architecture()
    draw_methodology()
