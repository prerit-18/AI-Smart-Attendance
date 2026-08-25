# System Architecture

This document describes the high-level architecture, processing pipelines, and data flows of the **AI Smart Attendance System**.

---

## 1. High-Level Flow Chart

The system operates via a continuous video processing loop integrated with a relational SQLite database and a recurrent temporal prediction module.

The main system flow follows:
```
[Camera Feed / Webcam]
         │
         ▼
[Face Detection (Haar Cascades)]
         │
         ▼
[Preprocessing & Quality Checks] (Filters out blur/small faces)
         │
         ▼
[CNN Embedding Model] (Extracts 128D features)
         │
         ▼
[Cosine Similarity Matching] (Compares with database templates)
         │
         ▼
[Confidence / Session Check] ──(Fails)──> [Log Unknown Attempt]
         │
      (Passes)
         │
         ▼
[Duplicate Prevention Guard] (Checks date + student_id + class session)
         │
         ▼
[Mark Attendance in SQLite]
         │
         ▼
[Streamlit UI / Dashboard Visualization]
```

At the same time, attendance records are processed through the forecasting sub-pipeline:
```
[Chronological Attendance Logs]
         │
         ▼
[Temporal sliding windows (W=5)]
         │
         ▼
[LSTM Recurrent Predictor]
         │
         ▼
[Future Class Presence Likelihoods]
```

---

## 2. Core Architectural Components

### A. Face Detection Layer (`src/face_detection.py`)
- Employs OpenCV's Haar Cascade classifier (`haarcascade_frontalface_default.xml`).
- Outputs coordinates `(x, y, w, h)` for all faces visible in the frame.
- Supports multi-face tracking in a single frame.

### B. Preprocessing Engine (`src/preprocessing.py`)
- **Dimensions**: Crops detected bounding boxes and resizes them to `160x160` pixels.
- **Normalization**: Scales RGB pixel channels from `[0, 255]` to `[-1.0, 1.0]`.
- **Blurriness Filter**: Computes the Laplacian variance. If the variance is below `100.0`, the frame is rejected to prevent blurry captures from corrupting database embeddings or producing false recognitions.

### C. Feature Extractor (`src/embeddings.py`)
- Encapsulates Keras architectures:
  - **MobileNetV2**: Uses pre-trained weights from ImageNet (transfer learning). Base layers are frozen, outputting a 128-dimensional projection vector.
  - **Custom CNN**: Implements a standard classification model with Batch Normalization and Dropout regularization layers (Course 2).
- Feature vectors are L2-normalized, projecting embeddings onto a unit hypersphere where Cosine Similarity is equivalent to a dot product.

### D. Matching & Verification (`src/face_recognition.py` / `src/attendance.py`)
- Compares active face embeddings with cached registered templates.
- **Decision Criteria**: Recognizes an identity if Cosine Similarity meets or exceeds the threshold (configured in `config.py`, default `0.6`).
- **Duplicate Prevention**: Prevents redundant logging for the same student within the same date-session slot.

### E. Relational Storage (`src/database.py`)
Uses local SQLite tables:
1. `students`: Stores demographic data and unique IDs.
2. `face_embeddings`: Relates Student IDs to serialized 128-dimensional embedding lists.
3. `attendance`: Stores daily attendance check-in logs.
4. `recognition_logs`: Audit trails logging successful and failed recognition events.

### F. Temporal Sequence Model (`src/sequence_model.py`)
- Restructures database check-ins into continuous binary vectors per student (e.g. `[1, 1, 0, 1, 1]`).
- Feeds rolling historical periods (sliding window `W=5`) into a recurrent LSTM network to forecast tomorrow's attendance likelihood.
