"""
train.py
--------
Updated Model Training Pipeline for Brain Tumor Detection.
Uses isolated Training (70%), Validation (15%), and Testing (15%) splits.
Evaluates ONLY the test dataset for final unbiased performance metrics.
Maintains exact ResNet50 architecture for full Grad-CAM++ backward compatibility.
"""

import os
import json
from datetime import datetime
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ---------------------------------------------------------------------------
# 1. Configuration & Directories
# ---------------------------------------------------------------------------
DATA_DIR = Path("data")
TRAIN_DIR = DATA_DIR / "Training"
VAL_DIR = DATA_DIR / "Validation"
TEST_DIR = DATA_DIR / "Testing"
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 3 # Fast demo / baseline; can be increased as needed
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]


def check_directories():
    """Ensure training, validation, and test directories are populated."""
    for d in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        if not d.exists() or len(list(d.glob("*/*"))) == 0:
            raise FileNotFoundError(
                f"Directory '{d}' is missing or empty. "
                "Please run 'python prepare_dataset.py' first to build the 70/15/15 dataset splits."
            )


def load_datasets():
    """Load isolated datasets with ResNet50 preprocessing."""
    print("\n[INFO] Loading datasets (Train 70%, Val 15%, Test 15%)...")
    
    train_ds = keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=123,
    )

    val_ds = keras.utils.image_dataset_from_directory(
        VAL_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    # Isolated test dataset: shuffle MUST be False for deterministic evaluation
    test_ds = keras.utils.image_dataset_from_directory(
        TEST_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    class_names = train_ds.class_names
    print(f"[INFO] Classes detected: {class_names}")

    # Standard ResNet50 Preprocessing
    preprocess_input = keras.applications.resnet50.preprocess_input

    def process_data(images, labels):
        return preprocess_input(images), labels

    train_ds = train_ds.map(process_data, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.map(process_data, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.map(process_data, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, test_ds, class_names


def build_model(num_classes):
    """
    Build transfer learning model with ResNet50.
    Preserves exact layer naming for Grad-CAM++ compatibility.
    """
    print("\n[INFO] Building ResNet50 Transfer Learning model...")
    base_model = keras.applications.ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3)
    )
    base_model.trainable = False

    inputs = keras.Input(shape=(224, 224, 3), name="input_layer_1")
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name="global_average_pooling2d")(x)
    x = layers.Dropout(0.2, name="dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="dense")(x)

    model = keras.Model(inputs, outputs, name="functional")

    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def plot_and_save_training_curves(history, output_dir):
    """Plot and save training vs validation accuracy and loss."""
    epochs_range = range(1, len(history.history["accuracy"]) + 1)

    # 1. Training & Validation Accuracy
    plt.figure(figsize=(7, 5))
    plt.plot(epochs_range, history.history["accuracy"], "b-o", label="Training Accuracy")
    if "val_accuracy" in history.history:
        plt.plot(epochs_range, history.history["val_accuracy"], "r--s", label="Validation Accuracy")
    plt.title("Model Accuracy across Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_dir / "training_accuracy.png", dpi=300)
    plt.close()

    # 2. Training & Validation Loss
    plt.figure(figsize=(7, 5))
    plt.plot(epochs_range, history.history["loss"], "b-o", label="Training Loss")
    if "val_loss" in history.history:
        plt.plot(epochs_range, history.history["val_loss"], "r--s", label="Validation Loss")
    plt.title("Model Loss across Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc="upper right")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_dir / "training_loss.png", dpi=300)
    plt.close()
    print(f"[SAVED] Training curves saved to '{output_dir}'")


def plot_confusion_matrix(cm, class_names, output_path):
    """Plot confusion matrix using matplotlib without external dependencies."""
    plt.figure(figsize=(7, 6))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix (Test Dataset)")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)

    # Normalize if possible for visual text coloring
    thresh = cm.max() / 2.0 if cm.max() > 0 else 1.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black"
            )

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[SAVED] Confusion matrix saved to '{output_path}'")


def evaluate_test_dataset(model, test_ds, class_names, output_dir):
    """
    Evaluate ONLY the test dataset to compute genuine, unbiased test performance metrics.
    No hardcoded values.
    """
    print("\n[INFO] Evaluating ONLY on the isolated Test Dataset...")
    eval_results = model.evaluate(test_ds, verbose=1)
    test_loss = float(eval_results[0])
    test_acc_eval = float(eval_results[1])

    y_true = []
    y_pred_probs = []

    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(labels.numpy())
        y_pred_probs.extend(preds)

    y_true = np.array(y_true)
    y_pred_probs = np.array(y_pred_probs)
    y_pred = np.argmax(y_pred_probs, axis=1)

    acc = float(accuracy_score(y_true, y_pred))
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    
    # Detailed text report
    report_text = classification_report(y_true, y_pred, target_names=class_names, digits=4, zero_division=0)
    report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)

    # 1. Save classification report text
    with open(output_dir / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write("BRAIN TUMOR MRI CLASSIFICATION - TEST EVALUATION REPORT\n")
        f.write("=" * 60 + "\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Test Accuracy: {acc * 100:.2f}%\n")
        f.write(f"Test Loss:     {test_loss:.4f}\n\n")
        f.write(report_text)
        f.write("\n\nConfusion Matrix:\n")
        f.write(np.array2string(cm, separator=", "))
        f.write("\n")

    # 2. Save confusion matrix plot
    plot_confusion_matrix(cm, class_names, output_dir / "confusion_matrix.png")

    # 3. Save metrics JSON
    metrics_data = {
        "timestamp": datetime.now().isoformat(),
        "test_accuracy": round(acc, 4),
        "test_loss": round(test_loss, 4),
        "weighted_precision": round(float(precision), 4),
        "weighted_recall": round(float(recall), 4),
        "weighted_f1": round(float(f1), 4),
        "macro_precision": round(float(macro_precision), 4),
        "macro_recall": round(float(macro_recall), 4),
        "macro_f1": round(float(macro_f1), 4),
        "classes": class_names,
        "confusion_matrix": cm.tolist(),
        "per_class": {
            cls: {
                "precision": round(float(report_dict[cls]["precision"]), 4),
                "recall": round(float(report_dict[cls]["recall"]), 4),
                "f1-score": round(float(report_dict[cls]["f1-score"]), 4),
                "support": int(report_dict[cls]["support"])
            }
            for cls in class_names if cls in report_dict
        }
    }

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=2)

    print("\n" + "=" * 55)
    print("FINAL TEST DATASET EVALUATION RESULTS")
    print("=" * 55)
    print(f"Accuracy:  {acc * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall:    {recall * 100:.2f}%")
    print(f"F1-Score:  {f1 * 100:.2f}%")
    print("=" * 55)
    print(f"[SAVED] Results exported to '{output_dir}/metrics.json' and 'classification_report.txt'")


def main():
    check_directories()
    train_ds, val_ds, test_ds, class_names = load_datasets()

    model = build_model(num_classes=len(class_names))

    print(f"\n[INFO] Starting training for {EPOCHS} epochs...")
    history = model.fit(
        train_ds,
        epochs=EPOCHS,
        validation_data=val_ds,
        verbose=1
    )

    # Save model weights preserving compatibility
    model_save_path = "brain_tumor_model.keras"
    print(f"\n[INFO] Saving trained model weights to '{model_save_path}'...")
    model.save(model_save_path)
    print(f"[SUCCESS] Model saved to '{model_save_path}'")

    # Save training curves
    plot_and_save_training_curves(history, RESULTS_DIR)

    # Evaluate exclusively on isolated test dataset
    evaluate_test_dataset(model, test_ds, class_names, RESULTS_DIR)


if __name__ == "__main__":
    main()
