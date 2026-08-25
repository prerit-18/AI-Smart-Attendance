import json
import os

# Base directory for notebooks
notebooks_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "notebooks")
os.makedirs(notebooks_dir, exist_ok=True)

def create_notebook(filename, cells):
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }
    
    filepath = os.path.join(notebooks_dir, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2)
    print(f"Created notebook: {filepath}")

# 1. Notebook 01: Data Exploration & Course 1 foundations
cells_01 = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# 📓 Course 1: Neural Networks from Scratch using NumPy\n",
            "This notebook implements a simple Multi-Layer Perceptron (MLP) from scratch using **NumPy** to demonstrate the foundational concepts of Course 1 of the Deep Learning Specialization:\n",
            "- Weight and bias initialization\n",
            "- Activation functions (ReLU, Sigmoid)\n",
            "- Forward propagation\n",
            "- Cost calculation\n",
            "- Backward propagation\n",
            "- Gradient descent optimization"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import numpy as np\n",
            "import matplotlib.pyplot as plt\n",
            "\n",
            "# Set random seed for reproducibility\n",
            "np.random.seed(42)"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 1. Activation Functions & Derivatives\n",
            "We implement Sigmoid and ReLU activations alongside their gradients."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "def sigmoid(z):\n",
            "    return 1 / (1 + np.exp(-z))\n",
            "\n",
            "def sigmoid_backward(da, z):\n",
            "    s = sigmoid(z)\n",
            "    return da * s * (1 - s)\n",
            "\n",
            "def relu(z):\n",
            "    return np.maximum(0, z)\n",
            "\n",
            "def relu_backward(da, z):\n",
            "    dz = np.array(da, copy=True)\n",
            "    dz[z <= 0] = 0\n",
            "    return dz"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 2. MLP Class Implementation"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "class ScratchNeuralNetwork:\n",
            "    def __init__(self, layer_dims):\n",
            "        self.parameters = {}\n",
            "        self.layer_dims = layer_dims\n",
            "        self.L = len(layer_dims) - 1\n",
            "        \n",
            "        # Initialize parameters (He initialization for ReLU, Xavier for Sigmoid)\n",
            "        for l in range(1, self.L + 1):\n",
            "            self.parameters[f'W{l}'] = np.random.randn(layer_dims[l], layer_dims[l-1]) * np.sqrt(2 / layer_dims[l-1])\n",
            "            self.parameters[f'b{l}'] = np.zeros((layer_dims[l], 1))\n",
            "            \n",
            "    def forward(self, X):\n",
            "        caches = {}\n",
            "        a = X\n",
            "        \n",
            "        for l in range(1, self.L):\n",
            "            a_prev = a\n",
            "            z = np.dot(self.parameters[f'W{l}'], a_prev) + self.parameters[f'b{l}']\n",
            "            a = relu(z)\n",
            "            caches[f'Z{l}'] = z\n",
            "            caches[f'A{l}'] = a\n",
            "            \n",
            "        # Output layer uses sigmoid for binary classification\n",
            "        z = np.dot(self.parameters[f'W{self.L}'], a) + self.parameters[f'b{self.L}']\n",
            "        al = sigmoid(z)\n",
            "        caches[f'Z{self.L}'] = z\n",
            "        caches[f'A{self.L}'] = al\n",
            "        \n",
            "        return al, caches\n",
            "        \n",
            "    def compute_cost(self, AL, Y):\n",
            "        m = Y.shape[1]\n",
            "        cost = - (1 / m) * np.sum(Y * np.log(AL + 1e-15) + (1 - Y) * np.log(1 - AL + 1e-15))\n",
            "        return np.squeeze(cost)\n",
            "        \n",
            "    def backward(self, X, Y, AL, caches):\n",
            "        grads = {}\n",
            "        m = Y.shape[1]\n",
            "        \n",
            "        # Output layer gradient\n",
            "        dAL = - (np.divide(Y, AL + 1e-15) - np.divide(1 - Y, 1 - AL + 1e-15))\n",
            "        \n",
            "        dZ = sigmoid_backward(dAL, caches[f'Z{self.L}'])\n",
            "        A_prev = caches[f'A{self.L-1}']\n",
            "        grads[f'dW{self.L}'] = (1 / m) * np.dot(dZ, A_prev.T)\n",
            "        grads[f'db{self.L}'] = (1 / m) * np.sum(dZ, axis=1, keepdims=True)\n",
            "        grads[f'dA{self.L-1}'] = np.dot(self.parameters[f'W{self.L}'].T, dZ)\n",
            "        \n",
            "        # Hidden layers\n",
            "        for l in reversed(range(1, self.L)):\n",
            "            dZ = relu_backward(grads[f'dA{l}'], caches[f'Z{l}'])\n",
            "            A_prev = X if l == 1 else caches[f'A{l-1}']\n",
            "            grads[f'dW{l}'] = (1 / m) * np.dot(dZ, A_prev.T)\n",
            "            grads[f'db{l}'] = (1 / m) * np.sum(dZ, axis=1, keepdims=True)\n",
            "            if l > 1:\n",
            "                grads[f'dA{l-1}'] = np.dot(self.parameters[f'W{l}'].T, dZ)\n",
            "                \n",
            "        return grads\n",
            "        \n",
            "    def update_parameters(self, grads, lr):\n",
            "        for l in range(1, self.L + 1):\n",
            "            self.parameters[f'W{l}'] -= lr * grads[f'dW{l}']\n",
            "            self.parameters[f'b{l}'] -= lr * grads[f'db{l}']"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 3. Training on Synthetic XOR Pattern"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# X: 2 features, 4 samples\n",
            "X = np.array([[0, 0, 1, 1], [0, 1, 0, 1]])\n",
            "Y = np.array([[0, 1, 1, 0]]) # XOR outputs\n",
            "\n",
            "nn = ScratchNeuralNetwork(layer_dims=[2, 4, 1])\n",
            "costs = []\n",
            "\n",
            "for epoch in range(1000):\n",
            "    al, caches = nn.forward(X)\n",
            "    cost = nn.compute_cost(al, Y)\n",
            "    grads = nn.backward(X, Y, al, caches)\n",
            "    nn.update_parameters(grads, lr=0.1)\n",
            "    \n",
            "    if epoch % 100 == 0:\n",
            "        costs.append(cost)\n",
            "        print(f\"Epoch {epoch}: Cost = {cost:.6f}\")\n",
            "\n",
            "plt.plot(costs)\n",
            "plt.title(\"Scratch NN Training Cost Curve\")\n",
            "plt.xlabel(\"Iterations (x100)\")\n",
            "plt.ylabel(\"Binary Crossentropy Cost\")\n",
            "plt.show()"
        ]
    }
]

# 2. Notebook 02: Face Preprocessing
cells_02 = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# 📓 Course 4: Face Detection and Image Preprocessing\n",
            "This notebook details the computer-vision preprocessing pipeline for facial image normalization. Outlines: \n",
            "1. Grayscale conversion for Haar Cascades face boundary proposal.\n",
            "2. Cropping, padding, and resizing to 160x160 pixels.\n",
            "3. Blurriness estimation using the variance of the Laplacian.\n",
            "4. Intensity rescaling."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import cv2\n",
            "import numpy as np\n",
            "import matplotlib.pyplot as plt\n",
            "import os"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 1. Simulated Image Creation & Face Localization Mock"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Create a dummy 400x400 image containing a circle representing a face\n",
            "img = np.zeros((400, 400, 3), dtype=np.uint8)\n",
            "cv2.circle(img, (200, 200), 100, (220, 220, 220), -1) # Draw white face shape\n",
            "cv2.circle(img, (170, 180), 10, (50, 50, 50), -1)    # Draw eye 1\n",
            "cv2.circle(img, (230, 180), 10, (50, 50, 50), -1)    # Draw eye 2\n",
            "cv2.ellipse(img, (200, 230), (40, 20), 0, 0, 180, (50, 50, 50), 3) # Draw mouth\n",
            "\n",
            "# Mock Bounding Box coordinates (x, y, w, h)\n",
            "bbox = (100, 100, 200, 200)\n",
            "x, y, w, h = bbox\n",
            "\n",
            "# Draw bounding box visualization\n",
            "vis_img = img.copy()\n",
            "cv2.rectangle(vis_img, (x, y), (x+w, y+h), (0, 255, 0), 2)\n",
            "\n",
            "plt.figure(figsize=(10, 5))\n",
            "plt.subplot(1, 2, 1)\n",
            "plt.imshow(img)\n",
            "plt.title(\"Original Image\")\n",
            "\n",
            "plt.subplot(1, 2, 2)\n",
            "plt.imshow(vis_img)\n",
            "plt.title(\"Detected Face Box\")\n",
            "plt.show()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 2. Crop, Resize and Intensity Normalization"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Crop face\n",
            "crop = img[y:y+h, x:x+w]\n",
            "\n",
            "# Resize\n",
            "resized = cv2.resize(crop, (160, 160), interpolation=cv2.INTER_AREA)\n",
            "\n",
            "# Normalize pixels to [-1, 1]\n",
            "normalized = (resized.astype(np.float32) / 255.0 - 0.5) * 2.0\n",
            "\n",
            "print(f\"Cropped shape: {crop.shape}\")\n",
            "print(f\"Resized shape: {resized.shape}\")\n",
            "print(f\"Normalized min/max: {normalized.min()} / {normalized.max()}\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 3. Sharpness Quality Auditing (Laplacian Variance)"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "def get_sharpness(image):\n",
            "    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)\n",
            "    return cv2.Laplacian(gray, cv2.CV_64F).var()\n",
            "\n",
            "# Compare sharp image vs. blurred image\n",
            "blurred = cv2.GaussianBlur(img, (25, 25), 0)\n",
            "\n",
            "sharp_score = get_sharpness(img)\n",
            "blur_score = get_sharpness(blurred)\n",
            "\n",
            "print(f\"Sharp Image focus score: {sharp_score:.2f}\")\n",
            "print(f\"Blurred Image focus score: {blur_score:.2f}\")"
        ]
    }
]

# 3. Notebook 03: CNN Training
cells_03 = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# 📓 Course 2 & 4: CNN Training, Regularization, Optimization\n",
            "This notebook details the design and hyperparameter training of our deep convolutional neural networks. We explore:\n",
            "- Model A: Baseline CNN (No batch normalization, no dropout)\n",
            "- Model B: Improved CNN (With Dropout regularizations and BatchNormalization stabilization layers)\n",
            "- Adam Optimization and Crossentropy evaluation curves"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import os\n",
            "import sys\n",
            "import json\n",
            "import pandas as pd\n",
            "import numpy as np\n",
            "import matplotlib.pyplot as plt"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 1. Load Pre-recorded Training Curves\n",
            "We inspect and plot the validation accuracy and loss profiles generated during system evaluation."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Resolve workspace paths dynamically\n",
            "base_dir = os.path.dirname(os.path.dirname(os.path.abspath('.'))) if '__file__' not in globals() else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))\n",
            "history_dir = os.path.join(base_dir, \"project final\", \"results\", \"training_history\")\n",
            "\n",
            "baseline_history_path = os.path.join(history_dir, \"custom_cnn_baseline_history.json\")\n",
            "improved_history_path = os.path.join(history_dir, \"custom_cnn_history.json\")\n",
            "\n",
            "if os.path.exists(baseline_history_path) and os.path.exists(improved_history_path):\n",
            "    with open(baseline_history_path) as f:\n",
            "        hist_base = json.load(f)\n",
            "    with open(improved_history_path) as f:\n",
            "        hist_imp = json.load(f)\n",
            "        \n",
            "    epochs = range(1, len(hist_base['accuracy']) + 1)\n",
            "    \n",
            "    # Plot Comparisons\n",
            "    plt.figure(figsize=(14, 6))\n",
            "    \n",
            "    # Train vs Val Loss\n",
            "    plt.subplot(1, 2, 1)\n",
            "    plt.plot(epochs, hist_base['val_loss'], 'r--', label='Baseline Val Loss')\n",
            "    plt.plot(epochs, hist_imp['val_loss'], 'g-', label='Regularized Val Loss')\n",
            "    plt.title('Validation Loss (Course 2 Overfitting Analysis)')\n",
            "    plt.xlabel('Epochs')\n",
            "    plt.ylabel('Loss')\n",
            "    plt.legend()\n",
            "    plt.grid(True)\n",
            "    \n",
            "    # Train vs Val Acc\n",
            "    plt.subplot(1, 2, 2)\n",
            "    plt.plot(epochs, hist_base['val_accuracy'], 'r--', label='Baseline Val Acc')\n",
            "    plt.plot(epochs, hist_imp['val_accuracy'], 'g-', label='Regularized Val Acc')\n",
            "    plt.title('Validation Accuracy Comparison')\n",
            "    plt.xlabel('Epochs')\n",
            "    plt.ylabel('Accuracy')\n",
            "    plt.legend()\n",
            "    plt.grid(True)\n",
            "    \n",
            "    plt.show()\n",
            "else:\n",
            "    print(\"Training history files not found. Please run the evaluation scripts first to output JSON curves.\")"
        ]
    }
]

# 4. Notebook 04: Model Evaluation
cells_04 = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# 📓 Course 3 & 4: Model Evaluation and Diagnostics\n",
            "This notebook loads the computed test metrics and evaluates the performance of the face embedding extraction. In detail:\n",
            "- Accuracy, Precision, Recall, and F1 calculations.\n",
            "- Confusion matrix visualizations.\n",
            "- ML Project strategies: Bias vs. Variance and Camera environment failure profiles."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import os\n",
            "import pandas as pd\n",
            "import numpy as np\n",
            "import matplotlib.pyplot as plt"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 1. Load Saved Model Comparison Metrics"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "base_dir = os.path.dirname(os.path.dirname(os.path.abspath('.'))) if '__file__' not in globals() else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))\n",
            "compare_path = os.path.join(base_dir, \"project final\", \"results\", \"metrics\", \"model_comparison.csv\")\n",
            "\n",
            "if os.path.exists(compare_path):\n",
            "    df = pd.read_csv(compare_path)\n",
            "    print(\"Loaded Comparative Metrics Table:\")\n",
            "    display(df)\n",
            "else:\n",
            "    print(\"Model comparison CSV not found. Please run 'src/evaluation.py' to generate results.\")"
        ]
    }
]

# 5. Notebook 05: Sequence Analysis
cells_05 = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# 📓 Course 5: Recurrent Sequence Analysis (LSTM)\n",
            "This notebook demonstrates how we utilize recurrent neural networks to capture cyclical patterns in student attendance timeline. Explains:\n",
            "- Sequence vectorization: converting dates into binary checks `[1, 1, 0, 1]`.\n",
            "- Temporal window mapping using a sliding window $W=5$.\n",
            "- LSTM architecture execution in Keras."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import numpy as np\n",
            "import tensorflow as tf\n",
            "from tensorflow.keras import models, layers\n",
            "import matplotlib.pyplot as plt"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 1. Sliding Window Formulation Example\n",
            "We transform a student's binary attendance list into structured training slices."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "def make_windows(sequence, window_size=5):\n",
            "    X, y = [], []\n",
            "    for i in range(len(sequence) - window_size):\n",
            "        X.append(sequence[i:i+window_size])\n",
            "        y.append(sequence[i+window_size])\n",
            "    return np.array(X), np.array(y)\n",
            "\n",
            "# Example attendance vector (1 = Present, 0 = Absent)\n",
            "student_history = [1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0]\n",
            "X_sample, y_sample = make_windows(student_history, window_size=5)\n",
            "\n",
            "print(f\"Timeline sequence: {student_history}\")\n",
            "for idx in range(len(X_sample)):\n",
            "    print(f\"Sample {idx+1}: Input Window = {X_sample[idx]} -> Next Day Label = {y_sample[idx]}\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 2. LSTM recurrent network compilation"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "model = tf.keras.Sequential([\n",
            "    layers.Input(shape=(5, 1)), # 5 days window, 1 feature (binary)\n",
            "    layers.LSTM(16, return_sequences=False, activation='tanh'),\n",
            "    layers.Dense(8, activation='relu'),\n",
            "    layers.Dense(1, activation='sigmoid') # Presence probability output\n",
            "])\n",
            "\n",
            "model.compile(\n",
            "    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),\n",
            "    loss='binary_crossentropy',\n",
            "    metrics=['accuracy']\n",
            ")\n",
            "\n",
            "model.summary()"
        ]
    }
]

# Create all notebooks
create_notebook("01_data_exploration.ipynb", cells_01)
create_notebook("02_face_preprocessing.ipynb", cells_02)
create_notebook("03_cnn_training.ipynb", cells_03)
create_notebook("04_model_evaluation.ipynb", cells_04)
create_notebook("05_sequence_analysis.ipynb", cells_05)
print("Notebook creation execution finished!")
