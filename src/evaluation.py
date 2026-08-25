import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import json

# Add base directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import METRICS_DIR, CONFUSION_MATRIX_DIR, PLOTS_DIR
from src.training import train_and_evaluate_cnn

def run_full_evaluation():
    """
    Trains (on synthetic/real data) and evaluates all three model options:
    1. custom_cnn_baseline (No Dropout/BN)
    2. custom_cnn (With Dropout + BN - Course 2 regularized)
    3. mobilenet (Course 4 Transfer Learning)
    Saves metrics and comparisons for the Model Evaluation Page.
    """
    models_to_evaluate = ["custom_cnn_baseline", "custom_cnn", "mobilenet"]
    results = []
    
    for m_type in models_to_evaluate:
        print(f"\n--- Training & Evaluating Model: {m_type} ---")
        # Run training for 5 epochs to get actual metrics quickly and verify pipeline
        trained_model, history, test_data = train_and_evaluate_cnn(model_type=m_type, epochs=5)
        X_test, y_test, label_encoder = test_data
        
        # Predict on test set
        y_pred_probs = trained_model.predict(X_test)
        y_pred = np.argmax(y_pred_probs, axis=1)
        y_true = np.argmax(y_test, axis=1)
        
        # Calculate overall accuracy
        accuracy = np.mean(y_pred == y_true)
        
        # Calculate Precision, Recall, F1 (weighted)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")
        
        # Generate Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        class_names = label_encoder.classes_
        
        plot_confusion_matrix(cm, class_names, m_type)
        
        # Record results
        notes = ""
        if m_type == "custom_cnn_baseline":
            notes = "Custom architecture without regularization (Course 2 baseline)"
        elif m_type == "custom_cnn":
            notes = "Custom architecture with Dropout (0.5) and BatchNormalization"
        elif m_type == "mobilenet":
            notes = "Transfer Learning feature extraction using pretrained MobileNetV2"
            
        results.append({
            "Model": m_type,
            "Accuracy": round(float(accuracy), 4),
            "Precision": round(float(precision), 4),
            "Recall": round(float(recall), 4),
            "F1": round(float(f1), 4),
            "Notes": notes
        })
        
    # Write comparison CSV
    df_compare = pd.DataFrame(results)
    os.makedirs(METRICS_DIR, exist_ok=True)
    comparison_path = os.path.join(METRICS_DIR, "model_comparison.csv")
    df_compare.to_csv(comparison_path, index=False)
    print(f"\nSaved model comparison table to: {comparison_path}")
    
    return df_compare

def plot_confusion_matrix(cm, classes, model_name):
    """
    Plots and saves confusion matrix heatmap.
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=classes, yticklabels=classes)
    plt.title(f"Confusion Matrix - {model_name.upper()}")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    
    os.makedirs(CONFUSION_MATRIX_DIR, exist_ok=True)
    cm_path = os.path.join(CONFUSION_MATRIX_DIR, f"{model_name}_confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()
    print(f"Saved confusion matrix plot to: {cm_path}")

if __name__ == "__main__":
    run_full_evaluation()
