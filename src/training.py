import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import json

# Add base directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import (
    STUDENTS_DIR, IMAGE_SIZE, FACE_MODEL_DIR, 
    TRAINING_HISTORY_DIR, PLOTS_DIR, METRICS_DIR
)
from src.preprocessing import resize_image, normalize_image
from src.embeddings import build_custom_cnn, build_mobilenet_extractor

def generate_synthetic_image_dataset(num_classes=5, images_per_class=20):
    """
    Generates a synthetic image dataset (random pixel noise with unique class shapes)
    to allow the CNN training pipeline to run out of the box.
    """
    print(f"Generating synthetic image dataset: {num_classes} classes, {images_per_class} images/class.")
    X = []
    y = []
    
    np.random.seed(42)
    for class_id in range(num_classes):
        # Create a "distinctive" signature shape for this class
        signature_shape = np.random.uniform(0.1, 0.9, (10, 10, 3))
        
        for img_id in range(images_per_class):
            # Base noise
            img = np.random.normal(127.0, 15.0, (IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
            
            # Draw class specific "signature patch" in the center to make it learnable
            h_start = (IMAGE_SIZE[0] - 10) // 2
            w_start = (IMAGE_SIZE[1] - 10) // 2
            img[h_start:h_start+10, w_start:w_start+10] = signature_shape * 255.0
            
            # Add random noise perturbations to simulate camera lighting
            brightness_shift = np.random.uniform(-30, 30)
            img = np.clip(img + brightness_shift, 0, 255).astype(np.uint8)
            
            X.append(img)
            y.append(f"ST{class_id+1:03d}")
            
    return np.array(X), np.array(y)

def load_real_dataset():
    """
    Loads face crop images from the data/students/ directory.
    Returns:
        X: numpy array of images
        y: numpy array of student_ids (labels)
    """
    X = []
    y = []
    
    if not os.path.exists(STUDENTS_DIR):
        return None, None
        
    student_folders = [f for f in os.listdir(STUDENTS_DIR) if os.path.isdir(os.path.join(STUDENTS_DIR, f)) and f.startswith("ST")]
    
    for student_id in student_folders:
        student_path = os.path.join(STUDENTS_DIR, student_id)
        for filename in os.listdir(student_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(student_path, filename)
                img = cv2.imread(img_path)
                if img is not None:
                    # Preprocess immediately
                    resized = resize_image(img, IMAGE_SIZE)
                    X.append(resized)
                    y.append(student_id)
                    
    if len(X) == 0:
        return None, None
        
    return np.array(X), np.array(y)

def train_and_evaluate_cnn(model_type="custom_cnn", epochs=10, batch_size=16):
    """
    Complete model training pipeline: loads data, splits, trains, evaluates, saves.
    """
    # 1. Load Data
    X_raw, y_raw = load_real_dataset()
    is_synthetic = False
    
    if X_raw is None or len(X_raw) < 10:
        print("No student registration images found or dataset too small. Falling back to synthetic face dataset.")
        X_raw, y_raw = generate_synthetic_image_dataset(num_classes=5, images_per_class=20)
        is_synthetic = True
        
    # 2. Preprocess images: scale to range [-1, 1]
    X = np.array([normalize_image(img) for img in X_raw])
    
    # Encode text labels into categories
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_raw)
    num_classes = len(np.unique(y_raw))
    y_cat = to_categorical(y_encoded, num_classes)
    
    # 3. Train/Val/Test Split (Course 3 methodology)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y_cat, test_size=0.15, random_state=42, stratify=y_raw
    )
    
    # Further split train/val
    # Stratify needs label indexes instead of hot vectors, so extract back argmax
    strat_labels = np.argmax(y_train_val, axis=1)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.20, random_state=42, stratify=strat_labels
    )
    
    print(f"Dataset split: Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    
    # 4. Build Model with classification head
    inputs = tf.keras.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
    
    if model_type == "custom_cnn":
        # Course 2 design: with Dropout + Batch Normalization
        base_extractor = build_custom_cnn(input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3), use_regularization=True)
    elif model_type == "custom_cnn_baseline":
        # Course 2 comparison baseline: without Dropout/BN
        base_extractor = build_custom_cnn(input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3), use_regularization=False)
    elif model_type == "mobilenet":
        # Course 4 transfer learning
        base_extractor = build_mobilenet_extractor(input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
        
    # Attach classification head for training
    embeddings = base_extractor(inputs)
    classification_out = tf.keras.layers.Dense(num_classes, activation="softmax")(embeddings)
    training_model = tf.keras.Model(inputs, classification_out)
    
    training_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    # 5. Train Model
    history = training_model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )
    
    # 6. Save Base Embedding Model (without classification head)
    # This matches our deployment pipeline which relies on cosine embedding distances
    os.makedirs(FACE_MODEL_DIR, exist_ok=True)
    save_filename = f"{model_type}_face_model_synthetic.h5" if is_synthetic else f"{model_type}_face_model.h5"
    base_extractor.save(os.path.join(FACE_MODEL_DIR, save_filename))
    
    # Save training history as JSON
    hist_dict = history.history
    history_path = os.path.join(TRAINING_HISTORY_DIR, f"{model_type}_history.json")
    with open(history_path, 'w') as f:
        json.dump(hist_dict, f)
        
    # 7. Generate and save Plots
    plot_training_curves(hist_dict, model_type)
    
    # 8. Evaluate on test set
    loss, acc = training_model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy for {model_type}: {acc * 100:.2f}%")
    
    # Save evaluation summary
    metrics_path = os.path.join(METRICS_DIR, f"{model_type}_test_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump({
            "model_type": model_type,
            "test_accuracy": float(acc),
            "test_loss": float(loss),
            "dataset_type": "synthetic" if is_synthetic else "real",
            "epochs": epochs
        }, f)
        
    return training_model, history.history, (X_test, y_test, label_encoder)

def plot_training_curves(history, model_name):
    """
    Plots training vs validation accuracy and loss. Saves plots to results/plots/.
    """
    epochs = range(1, len(history['accuracy']) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['accuracy'], 'bo-', label='Training Acc')
    plt.plot(epochs, history['val_accuracy'], 'ro-', label='Validation Acc')
    plt.title(f'{model_name.upper()} Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['loss'], 'bo-', label='Training Loss')
    plt.plot(epochs, history['val_loss'], 'ro-', label='Validation Loss')
    plt.title(f'{model_name.upper()} Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plot_path = os.path.join(PLOTS_DIR, f"{model_name}_training_curves.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved training curves plot to: {plot_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train face recognition model.")
    parser.add_argument("--model", type=str, default="custom_cnn", choices=["custom_cnn", "custom_cnn_baseline", "mobilenet"])
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()
    
    train_and_evaluate_cnn(model_type=args.model, epochs=args.epochs)
