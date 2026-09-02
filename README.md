# 🎓 AI Smart Attendance System

> An AI-powered attendance management system combining real-time face
> recognition, deep learning, biometric embeddings, attendance
> analytics, and LSTM-based forecasting in a Streamlit dashboard.

[![Python](https://img.shields.io/badge/Python-3.9--3.12-blue?logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit)](https://streamlit.io/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green?logo=opencv)](https://opencv.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep%20Learning-orange?logo=tensorflow)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-Academic-lightgrey)](#license)

## 🔗 Repository

**GitHub:** https://github.com/prerit-18/AI-Smart-Attendance

------------------------------------------------------------------------

## 📌 Overview

AI Smart Attendance automates student attendance using computer vision
and deep learning.

The system provides a complete workflow for:

-   👤 Student registration
-   📷 Face detection and quality filtering
-   🧠 Deep-learning face embeddings
-   🔎 Face verification using cosine similarity
-   ✅ Automatic attendance marking
-   🗃️ Student and attendance database management
-   📊 Attendance dashboards and analytics
-   📈 LSTM-based attendance forecasting
-   🎯 Model comparison and evaluation
-   🔬 Demo mode for environments without a webcam

The project was developed as an academic capstone and as a practical
application of concepts from the Deep Learning Specialization, including
neural networks, optimization, CNNs, machine-learning project
structuring, and sequence models.

------------------------------------------------------------------------

## ✨ Key Features

### 👤 Student Registration

-   Register students through the Streamlit interface.
-   Capture and store face samples locally.
-   Generate model-specific facial embeddings.
-   Support automatic enrollment in Demo Mode using synthetic facial
    signatures.

### 🎥 Real-Time Face Recognition

-   Webcam-based face detection using OpenCV Haar Cascade.
-   Laplacian variance check for image-quality/blur filtering.
-   Face crop and resize to `160 × 160`.
-   MobileNetV2-based feature extraction.
-   L2-normalized face embeddings.
-   Cosine-similarity matching with a configurable recognition
    threshold.
-   Duplicate-attendance prevention.

### 🧠 Multiple Face Models

The application supports model selection from the Streamlit sidebar:

  Model              Purpose
  ------------------ -------------------------------------------------
  **MobileNetV2**    Transfer-learning based face feature extraction
  **Improved CNN**   Custom trained CNN
  **Baseline CNN**   Baseline CNN for comparison

**Important:** Face embeddings are model-specific. Registration and
recognition should use the same active model.

### 📊 Attendance Management

-   Dashboard with attendance statistics.
-   Student database management.
-   Attendance records and history.
-   Automatic duplicate prevention.
-   Student deletion with associated local biometric-data cleanup.

### 📈 Analytics & Forecasting

Historical attendance records are converted into chronological sequences
and passed to an LSTM model to forecast future attendance/class-presence
behavior.

The sequence pipeline uses a sliding window of `W = 5`.

### 🎯 Model Evaluation

The evaluation workflow compares the implemented deep-learning models
using: - Accuracy - Loss curves - Confusion matrices - Test-set
evaluation - Error analysis

### 🔬 Demo Mode

Demo Mode allows the application to be explored without a physical
webcam.

It can simulate: - Student enrollment - Facial signatures - Face
detection - Recognition/matching - Attendance marking

This is especially useful for cloud-hosted or remote demonstrations.

------------------------------------------------------------------------

## 🏗️ System Architecture

### 1. Face Recognition Pipeline

``` text
Webcam / Demo Input
        ↓
Haar Cascade Face Detection
        ↓
Image Quality / Blur Check
        ↓
Face Crop & Resize (160×160)
        ↓
MobileNetV2 / CNN Feature Extractor
        ↓
L2 Normalization
        ↓
Cosine Similarity Matching
        ↓
Recognition Threshold
        ↓
Duplicate Prevention
        ↓
SQLite Attendance Record
```

### 2. Attendance Forecasting Pipeline

``` text
SQLite Attendance Records
        ↓
Chronological Attendance Vectors
        ↓
Sliding Window (W=5)
        ↓
LSTM Sequence Model
        ↓
Dense + Sigmoid Output
        ↓
Future Attendance Forecast
```

------------------------------------------------------------------------

## 🛠️ Tech Stack

  Category                  Technologies
  ------------------------- -----------------------------
  Language                  Python
  Web UI                    Streamlit
  Computer Vision           OpenCV
  Deep Learning             TensorFlow / Keras
  Face Feature Extraction   MobileNetV2
  Sequence Modeling         LSTM
  Data Processing           NumPy, Pandas
  Database                  SQLite
  Visualization             Matplotlib, Seaborn, Plotly
  ML Utilities              Scikit-learn, Joblib
  Testing                   Pytest

------------------------------------------------------------------------

## 📂 Project Structure

``` text
AI-Smart-Attendance/
│
├── app.py
├── requirements.txt
├── .env.example
├── .gitignore
│
├── config/
│
├── data/
│   └── students/
│
├── models/
│
├── notebooks/
│
├── results/
│
├── scratch/
│
├── docs/
│
├── tests/
│
└── src/
    ├── analytics.py
    ├── attendance.py
    ├── database.py
    ├── draw_diagrams.py
    ├── embeddings.py
    ├── evaluation.py
    ├── face_detection.py
    ├── face_recognition.py
    ├── preprocessing.py
    ├── sequence_model.py
    ├── training.py
    │
    └── pages/
        ├── analytics.py
        ├── attendance_records.py
        ├── dashboard.py
        ├── live_attendance.py
        ├── model_evaluation.py
        ├── register_student.py
        └── student_database.py
```

------------------------------------------------------------------------

## 🚀 Getting Started

### Prerequisites

-   Python `3.9`--`3.12`
-   Webcam for live recognition
-   Git
-   Recommended: Python virtual environment

> A webcam is not required when using Demo Mode.

### 1. Clone the Repository

``` bash
git clone https://github.com/prerit-18/AI-Smart-Attendance.git
cd AI-Smart-Attendance
```

### 2. Create a Virtual Environment

#### macOS / Linux

``` bash
python3 -m venv .venv
source .venv/bin/activate
```

#### Windows CMD

``` cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

#### Windows PowerShell

``` powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

``` bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run the Application

``` bash
streamlit run app.py
```

The Streamlit dashboard will open in your browser.

------------------------------------------------------------------------

## 🔬 Using Demo Mode

If you do not have access to a webcam:

1.  Start the application.
2.  Enable **🔬 Enable Demo Mode** from the sidebar.
3.  Open **👤 Register Student**.
4.  Enter the student details.
5.  Use **Simulate Automatic Enrollment**.
6.  Open **🎥 Live Attendance**.
7.  Select a registered student.
8.  Use **Simulate Detection** to test the recognition and attendance
    workflow.

Demo Mode is intended for testing and showcasing the application's logic
without physical camera hardware.

------------------------------------------------------------------------

## 🎥 Using Live Attendance

For real webcam-based recognition:

1.  Start the application.
2.  Keep **Demo Mode** disabled.
3.  Register students using the **Register Student** page.
4.  Select the face model from the sidebar.
5.  Capture/register face samples.
6.  Open **Live Attendance**.
7.  Start the webcam.
8.  The system detects faces and compares their embeddings with
    registered students.
9.  Matching identities are recorded in the SQLite attendance database.

For consistent results, register and recognize students with the same
selected face model.

------------------------------------------------------------------------

## 🧪 Training & Evaluation

To run the model evaluation pipeline:

``` bash
python3 src/evaluation.py
```

The evaluation workflow compiles and evaluates the available models,
generates performance visualizations, and produces confusion matrices.

Results can then be explored from:

**🎯 Model Evaluation**

### Run Tests

``` bash
pytest -v
```

Tests cover important application logic such as database rules,
similarity calculations, and duplicate-attendance checks.

------------------------------------------------------------------------

## ⚙️ Configuration

The repository includes `.env.example` for configuration.

The default database configuration is:

``` text
DATABASE_PATH=database/attendance.db
```

The project also supports a configurable cosine-similarity recognition
threshold.

Copy `.env.example` to `.env` if you need local environment
configuration:

``` bash
cp .env.example .env
```

On Windows, create the `.env` file manually if required.

------------------------------------------------------------------------

## 💾 Data Storage

By default, the application is designed around **local storage**.

Typical local data includes:

``` text
database/
└── attendance.db

data/
└── students/
    └── <student-data>
```

Face samples, embeddings, and attendance records are intended to remain
on the local machine.

### ⚠️ Streamlit Cloud Deployment

If the application is deployed on Streamlit Community Cloud or another
ephemeral container environment, **do not treat the local SQLite
database or uploaded biometric files as permanent storage**.

For a persistent production deployment, replace the local storage layer
with managed services such as:

-   PostgreSQL / MongoDB for structured records
-   Object storage for images
-   Secure secret/environment-variable management
-   Persistent model/data storage

The current repository is primarily intended for local execution and
academic/demo deployment.

------------------------------------------------------------------------

## 🔐 Privacy & Responsible Use

This project processes biometric information and should be treated
accordingly.

The current design includes:

-   Local biometric-data storage.
-   No requirement to send face data to an external recognition API.
-   Student deletion with associated local data cleanup.
-   A prototype/academic-use warning.

### Important

This project is **not intended to be used as an institutional biometric
system without appropriate consent, security controls, legal review,
retention policies, and a formal biometric/privacy impact assessment**.

Do not commit: - Student face images - Face embeddings - SQLite
databases containing personal information - `.env` files containing
secrets - Any other personally identifiable or sensitive data

Use `.gitignore` to prevent sensitive files from being committed.

------------------------------------------------------------------------

## 📚 Deep Learning Concepts Demonstrated

  Concept                  Implementation
  ------------------------ -------------------------------------------
  Neural Networks          Custom CNN experimentation
  Optimization             Dropout, Batch Normalization, Adam
  ML Project Structuring   Train/test evaluation and error analysis
  CNNs                     Face representation and transfer learning
  Transfer Learning        MobileNetV2 feature extraction
  Similarity Learning      L2 normalization + cosine similarity
  Sequence Models          LSTM-based attendance forecasting
  Model Evaluation         Accuracy, loss curves, confusion matrices

------------------------------------------------------------------------

## 🧭 Application Pages

The Streamlit application currently provides:

-   📊 **Dashboard**
-   👤 **Register Student**
-   👥 **Student Database**
-   🎥 **Live Attendance**
-   📋 **Attendance Records**
-   📈 **Analytics & Sequence**
-   🎯 **Model Evaluation**
-   ℹ️ **About Project**

------------------------------------------------------------------------

## 🔮 Future Improvements

Potential next steps include:

-   Persistent cloud database integration
-   Secure cloud/object storage for biometric images
-   Authentication and role-based access
-   Face anti-spoofing / liveness detection
-   Better low-light and pose robustness
-   Multi-camera support
-   Real-time notifications
-   Advanced attendance anomaly detection
-   Containerized deployment
-   Automated CI/CD and test coverage
-   Privacy-preserving biometric storage

------------------------------------------------------------------------

## 📸 Screenshots

Add application screenshots here to make the repository easier to
understand:

``` text
docs/
├── dashboard.png
├── registration.png
├── live-attendance.png
├── attendance-records.png
├── analytics.png
└── model-evaluation.png
```

Then embed them in this README using:

``` markdown
![Dashboard](docs/dashboard.png)
![Live Attendance](docs/live-attendance.png)
```

------------------------------------------------------------------------

## 🧑‍💻 Author

**Prerit Mehta**

GitHub: https://github.com/prerit-18

------------------------------------------------------------------------

## 📄 License

This project is an academic/educational project. Add a formal
open-source license such as MIT before presenting the repository as an
open-source project.

------------------------------------------------------------------------

## ⚠️ Disclaimer

This system is developed for **educational, research, and demonstration
purposes**. Face recognition and biometric attendance involve sensitive
personal data. Any real-world deployment should implement appropriate
consent, security, privacy, data-retention, and regulatory safeguards.
