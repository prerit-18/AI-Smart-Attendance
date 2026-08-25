# 🎓 AI Smart Attendance System Using Deep Learning

An end-to-end, production-quality academic capstone project integrating **Computer Vision (CNNs)** and **Sequence Modeling (LSTMs)** to build an automated biometric attendance logging and predictive analysis system.

This project is structured as the practical application of concepts covered in the **Deep Learning Specialization**:
1. **Neural Networks and Deep Learning** (Scratch MLP exploration)
2. **Improving Deep Neural Networks** (Hyperparameter Tuning, Regularization, Dropout, Batch Normalization)
3. **Structuring Machine Learning Projects** (Train/Dev/Test partitions, Error Analysis)
4. **Convolutional Neural Networks** (Face embedding representation, transfer learning, cosine similarity matching)
5. **Sequence Models** (Chronological attendance forecasting using LSTMs)

---

## 🏗️ System Architecture & Workflow

The architecture is split into two primary pipelines:

1. **Biometric Face Recognition & Verification Pipeline (Real-Time)**:
   ```
   Webcam Capture ➔ Haar Cascade Face Detector ➔ Quality Filter (Laplacian blur variance check) 
   ➔ Crop/Resize (160x160) ➔ MobileNetV2 Feature Extractor ➔ L2 Normalization 
   ➔ Cosine Similarity Matrix Dot-Product ➔ Threshold Check (>= 0.6) 
   ➔ Duplicate Prevention Guard ➔ SQLite Mark Present.
   ```
2. **Temporal Attendance Forecasting Pipeline (Recurrent)**:
   ```
   SQLite Attendance Records ➔ Chronological Vector Ingestion ➔ Sliding Window Slicing (W=5)
   ➔ LSTM Sequence Layer ➔ Dense Sigmoid Output ➔ Future Class Presence Forecast.
   ```

---

## 🗂️ Project Structure

```
ai-smart-attendance/
│
├── app.py                     # Streamlit web app main entry point
├── requirements.txt           # Project library dependencies
├── README.md                  # Comprehensive project documentation
├── .gitignore                 # Files excluded from git tracking
├── .env.example               # Environmental configuration template
│
├── config/
│   └── config.py              # Centralized configuration settings and paths
│
├── data/
│   ├── raw/                   # Unprocessed imagery directory
│   ├── processed/             # Normalised cropped faces
│   └── students/              # Registered student face directories (gitignored)
│
├── models/
│   ├── face_model/            # Saved Keras CNN model weights (.h5)
│   ├── embeddings/            # Cached enrollment representations
│   └── sequence_model/        # Saved Keras LSTM model weights (.h5)
│
├── notebooks/
│   ├── 01_data_exploration.ipynb   # Neural network from scratch (NumPy MLP)
│   ├── 02_face_preprocessing.ipynb # Image cropping, resizing and focus checks
│   ├── 03_cnn_training.ipynb       # CNN training curves and optimization
│   ├── 04_model_evaluation.ipynb   # Test partition statistics and evaluation
│   └── 05_sequence_analysis.ipynb  # Recurrent windowing and LSTM training
│
├── src/
│   ├── __init__.py
│   ├── database.py            # SQLite operations, parameterized queries, cascades
│   ├── face_detection.py      # OpenCV Haar Cascade wrapper
│   ├── face_recognition.py    # Embedding comparisons and matching decisions
│   ├── embeddings.py          # Custom CNN & MobileNetV2 architecture graphs
│   ├── preprocessing.py       # Focus measurements and color rescalers
│   ├── training.py            # Classification head compile and train loop
│   ├── evaluation.py          # Precision/Recall comparisons and matrices
│   ├── attendance.py          # Attendance check-in and duplicate guards
│   ├── analytics.py           # SQL grouping aggregates for trends
│   ├── sequence_model.py      # Keras LSTM pipeline and synthetic logs generator
│   └── draw_diagrams.py       # Utility drawing system diagrams
│
├── pages/
│   ├── dashboard.py           # Visual KPI metrics and trend line charts
│   ├── register_student.py    # Biometric capture and registration
│   ├── live_attendance.py     # Live camera streams and face boxes
│   ├── attendance_records.py  # Logs filtering and CSV downloads
│   ├── analytics.py           # Section statistics and LSTM forecasters
│   └── model_evaluation.py    # CNN learning history and matrices plots
│
├── docs/
│   ├── system_architecture.png# Drawn architecture flowchart
│   ├── methodology.png        # Drawn methodology flow steps
│   ├── architecture.md        # Architecture text documentation
│   ├── methodology.md         # Methodology stages breakdown
│   └── course_mapping.md      # Course curriculum mapping details
│
└── tests/
    ├── test_database.py       # SQLite database validation suite
    ├── test_attendance.py     # Attendance check-in and duplicates tests
    └── test_recognition.py    # Similarity calculations test suite
```

---

## 🛠️ Installation & Setup

This application has been developed to run cross-platform on **macOS**, **Windows**, and **Linux**.

### Prerequisites
- Python `3.9` to `3.12` installed.
- Integrated or USB Webcam (only for live webcam mode; not needed for Demo Mode).

### Step 1: Clone and Navigate to Directory
```bash
git clone <repository-url>
cd project_final
```

### Step 2: Initialize Virtual Environment
- **On macOS/Linux**:
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```
- **On Windows (Command Prompt)**:
  ```cmd
  python -m venv .venv
  .venv\Scripts\activate.bat
  ```
- **On Windows (PowerShell)**:
  ```powershell
  python -m venv .venv
  .venv\Scripts\Activate.ps1
  ```

### Step 3: Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🚀 Running the Application

### Starting the Web Dashboard
Launch the Streamlit portal:
```bash
streamlit run app.py
```

### 🔬 Exploring in Demo Mode
If you are running on a virtual machine, remote server, or do not have access to a physical webcam:
1. Turn on the **🔬 Enable Demo Mode** checkbox in the Streamlit sidebar.
2. Under the **👤 Register Student** tab, enter details and click **Simulate Automatic Enrollment** (generates synthetic facial signatures).
3. Under the **🎥 Live Attendance** tab, choose a student from the dropdown list and click **Simulate Detection** (runs the matching logic, checks similarity against database vectors, manages duplicates, and saves attendance).

---

## 🧪 Model Training and Evaluation

To reproduce the deep learning results and generate metrics:

### Run Training and Evaluations
Execute the comparative script which compiles the three models, evaluates them on a test split, saves loss/accuracy curves, and plots confusion matrices:
```bash
python3 src/evaluation.py
```
After executing, navigate to the **🎯 Model Evaluation** page inside the Streamlit app to view comparison charts and error analysis tables.

### Run Automated Unit Tests
To verify database rules, similarity equations, and duplicate check gates, run pytest:
```bash
pytest -v
```

---

## 🛡️ Responsible Design & Biometric Privacy Safeguards

Biometric datasets require careful data management. This project applies privacy-by-design standards:
- **Local Storage**: All captured facial crops, serialized JSON embeddings, and SQLite records remain stored strictly on the local drive. No data is sent to external cloud APIs.
- **Biometric Erasure**: When a student is deleted from the **Manage Students** tab, cascade database deletions purge their SQL profiles, and the application recursively deletes their image directories from `data/students/`.
- **Prototype Warning**: This system is designed as an academic demonstration and should not be deployed in institutional or commercial environments without formal consent and a Biometric Impact Assessment.

---

## 👨‍💻 Author
- **Academic Capstone Project**
- Developed as part of B.Tech CSE-AIML Industrial Training.
# AI-Smart-Attendance
