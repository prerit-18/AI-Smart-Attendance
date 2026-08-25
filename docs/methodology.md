# Project Methodology

This document outlines the systematic engineering methodology used to design, train, evaluate, and operate the **AI Smart Attendance System**.

---

## 1. Stages of the Development Lifecycle

The system development follows 10 distinct phases:

### Stage 1: Data Collection & Enrollment
- Multiple webcam samples (15-20 frames) are captured per student.
- Facial angles (frontal, slight left/right tilts) and expressions (neutral, smile) are recorded to ensure robustness against face posture variations.

### Stage 2: Face Detection (Region Proposal)
- OpenCV Haar Cascades scan the captured image at multi-scales.
- Detections are localized to coordinate boxes `(x, y, w, h)`.
- If zero or multiple faces are detected, the capture is rejected to keep data clean.

### Stage 3: Preprocessing & Color Correction
- The localized face is cropped out from the background frame.
- Image is converted to RGB format.
- Crop dimension is standardized to `160x160` pixels using area interpolation.
- Pixels are normalized to `[-1, 1]` scaling range.

### Stage 4: Quality Filtering & Sharpness Verification
- Evaluates blurriness by computing the variance of the Laplacian:
  $$\text{Blur Score} = \text{Var}(\nabla^2 I)$$
- Only crops exceeding the minimum score (default `100.0`) are processed.

### Stage 5: Feature Extraction (Backbone Forward Pass)
- Preprocessed face matrices are passed through deep convolutional networks:
  - MobileNetV2 pretrained base outputs raw convolutional features.
  - Features are pooled and projected into a dense projection layer of 128 units.
- Output activation is L2-normalized onto a unit hypersphere.

### Stage 6: Template Registration
- The extracted 128D embedding vector is averaged across all captured samples.
- The resulting vector template is serialized as JSON and stored in SQLite under the student's unique ID.

### Stage 7: Similarity Verification (Cosine Distance)
- During live sessions, the incoming face's embedding $\mathbf{u}$ is matched against a database template $\mathbf{v}$ using Cosine Similarity:
  $$\text{Similarity}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2}$$
- Since vectors are L2-normalized ($\|\mathbf{u}\|_2 = \|\mathbf{v}\|_2 = 1$), this reduces to a simple dot product:
  $$\text{Similarity}(\mathbf{u}, \mathbf{v}) = \mathbf{u} \cdot \mathbf{v}$$

### Stage 8: Duplicate Prevention & Marking
- If the highest similarity score exceeds the threshold $T = 0.6$, the student's identity is verified.
- The system checks if an entry exists in the `attendance` table for the composite index `(student_id, date, session)`.
- If not, check-in is logged.

### Stage 9: Comparative Evaluation & Fine-Tuning
- Implements validation partitions to compare custom CNNs (with vs. without batch normalization and dropout) against Transfer Learning.
- Confusion matrices, accuracy curves, and error audits are logged.

### Stage 10: Sequence Prediction (Temporal Analytics)
- Attendance records are compiled chronologically.
- A sliding temporal window of size $W=5$ is structured.
- An LSTM network processes sequence windows to output Tomorrow's attendance likelihood.
