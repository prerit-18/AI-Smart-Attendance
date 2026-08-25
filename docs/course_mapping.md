# Deep Learning Specialization Course Mapping

This document provides a detailed mapping between the academic concepts learned in the five courses of the **Deep Learning Specialization (Coursera / DeepLearning.AI)** and their practical application inside the **AI Smart Attendance System**.

## Course Mapping Table

| Course / Specialization Module | Key Core Concepts | Project Implementation / Application |
|---|---|---|
| **Course 1: Neural Networks & Deep Learning** | Neurons, weights, bias, activation functions (Sigmoid, ReLU), forward propagation, cost functions, backpropagation, gradient descent. | Evaluated in `notebooks/01_data_exploration.ipynb` with a scratch NumPy-based multi-layer perceptron. Foundation of neural network mapping. |
| **Course 2: Improving Deep Neural Networks** | Hyperparameter tuning, Train/Val/Test splits, Regularization, Dropout, Batch Normalization, Adam optimizer, learning rate decay. | Implemented in `src/embeddings.py` (Custom CNN configuration). Compares regularized (Dropout + BN) vs. baseline models to mitigate overfitting. |
| **Course 3: Structuring Machine Learning Projects** | Dataset strategy, error analysis, bias/variance profile, false positives/negatives, diagnostic auditing, lighting/angle issues. | Handled in `pages/model_evaluation.py` and `notebooks/04_model_evaluation.ipynb`. Houses an Error Analysis audit table for webcam failures. |
| **Course 4: Convolutional Neural Networks** | Convolutions, pooling layers, Dense projections, Transfer learning, One-shot learning, Face embeddings, Cosine similarity. | Core recognition engine in `src/embeddings.py` and `src/face_recognition.py`. Loads pre-trained MobileNetV2 backbone to extract L2-normalized 128D feature vectors. |
| **Course 5: Sequence Models** | Recurrent Neural Networks (RNNs), LSTMs, GRUs, sliding temporal windows, sequence classification, binary state forecasting. | Analytical forecasting module in `src/sequence_model.py`. An LSTM network processes student attendance history vectors to calculate next-class presence chance. |

---

## Detailed Implementation Rationale

### Course 1: Foundations
* **Activation Functions**: We use ReLU activations for intermediate convolutional layers to combat vanishing gradients, and Sigmoid for binary prediction outputs in our sequence LSTM model.
* **Loss Functions**: Categorical crossentropy is used during face classification training, and binary crossentropy is used for the LSTM next-day attendance prediction.

### Course 2: Optimization & Regularization
* **Batch Normalization**: Applied after each Conv2D layer in our custom CNN to stabilize activation distribution across mini-batches, allowing faster learning rates.
* **Dropout**: A rate of `0.5` is added to the dense layers to randomly disable neurons during training, forcing the network to learn redundant representations and prevent overfitting on small student face datasets.
* **Adam Optimizer**: Used as the default optimizer because it computes adaptive learning rates for individual parameters using estimates of first and second moments of the gradients.

### Course 3: ML Strategy & Error Analysis
* **Data Splitting**: Dataset partitioning uses `85%` for training/validation (split as `80%` train, `20%` val) and `15%` strictly for testing. This ensures generalization metrics are unbiased.
* **Auditing failures**: Real world cameras present lighting, blur, and angle challenges. Preprocessing controls are established to filter out small or blurry bounding boxes before embedding extraction occurs.

### Course 4: CNNs & Face Embeddings
* **Transfer Learning**: Rather than training a deep facial model from scratch on small samples, we utilize a MobileNetV2 backbone pre-trained on ImageNet to extract general edge, shape, and contrast descriptors.
* **L2-Normalized Space**: Normalizing the embedding vector to a unit sphere ensures that the Euclidean distance matches cosine similarity, simplifying database matches to a fast matrix dot-product.

### Course 5: Recurrent Sequence Analytics
* **Sequence Modeling**: Attendance is a chronological record. We transform attendance logs into periodic histories (e.g. `[1.0, 1.0, 0.0, 1.0, 1.0]`).
* **LSTM Processing**: The LSTM recurrent cells retain memory across dates, capturing temporal dependencies such as lower attendance on Fridays or cyclic absent patterns.
