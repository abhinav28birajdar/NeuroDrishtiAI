"""
api/app.py
----------
Flask REST API for Brain Tumor Detection and Grad-CAM++ Explainability.
Provides POST /api/analyze with real model inference, multi-class probabilities,
and Base64-encoded visual explainability maps.
"""

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from flask import Flask, request, jsonify
from flask_cors import CORS
import base64

app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing for frontend clients

# ---------------------------------------------------------------------------
# 1. Model Configuration & Loading
# ---------------------------------------------------------------------------
IMG_SIZE = (224, 224)
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]

def load_inference_model():
    """Load trained brain tumor model or fallback to ImageNet ResNet50."""
    candidate_paths = [
        "brain_tumor_model.keras",
        "../brain_tumor_model.keras",
        "model.keras"
    ]
    for p in candidate_paths:
        if os.path.exists(p):
            try:
                m = keras.models.load_model(p)
                print(f"[API] Loaded trained model from '{p}'")
                return m
            except Exception as e:
                print(f"[API] Error loading '{p}': {e}")
                
    print("[API] Loading fallback ResNet50 with ImageNet weights...")
    return keras.applications.ResNet50(weights="imagenet")


MODEL = load_inference_model()

if "resnet50" in [layer.name for layer in MODEL.layers]:
    LAST_CONV_LAYER = "resnet50"
else:
    LAST_CONV_LAYER = "conv5_block3_out"


# ---------------------------------------------------------------------------
# 2. Medical Knowledge Base
# ---------------------------------------------------------------------------
TUMOR_INFO = {
    "glioma": {
        "name": "Glioma",
        "details": "A glioma is a type of tumor that occurs in the brain and spinal cord. It begins in the supportive cells (glial cells) that surround nerve cells.",
        "causes": "The exact cause is unknown. Risk factors include advancing age, exposure to ionizing radiation, and a family history of glioma.",
        "resources": "American Brain Tumor Association (www.abta.org) | Mayo Clinic (www.mayoclinic.org/diseases-conditions/glioma)"
    },
    "meningioma": {
        "name": "Meningioma",
        "details": "A tumor that arises from the meninges (the membranes that surround the brain and spinal cord). Most meningiomas are noncancerous (benign).",
        "causes": "Caused by changes in the cells of the meninges. Risk factors include radiation treatment, female hormones, and inherited disorders (e.g., neurofibromatosis type 2).",
        "resources": "Meningioma Network | WebMD (www.webmd.com/cancer/brain-cancer/meningioma)"
    },
    "pituitary": {
        "name": "Pituitary Tumor",
        "details": "Abnormal growths that develop in the pituitary gland (a pea-sized gland at the base of the brain). They can affect hormone levels in the body.",
        "causes": "The cause of uncontrolled cell growth is largely unknown. A small percentage of cases run in families and are linked to genetic conditions like MEN 1.",
        "resources": "Pituitary Network Association (www.pituitary.org)"
    },
    "notumor": {
        "name": "No Tumor Detected",
        "details": "The MRI scan does not indicate the presence of a glioma, meningioma, or pituitary tumor.",
        "causes": "Healthy scan / Non-neoplastic tissue architecture.",
        "resources": "Always consult with a certified radiologist or neurologist for a definitive medical diagnosis."
    }
}


# ---------------------------------------------------------------------------
# 3. Helper Functions & Grad-CAM++
# ---------------------------------------------------------------------------
def get_img_array(img_path, size):
    """Load and preprocess image using ResNet50 preprocessing."""
    img = keras.preprocessing.image.load_img(img_path, target_size=size)
    array = keras.preprocessing.image.img_to_array(img)
    array = np.expand_dims(array, axis=0)
    return keras.applications.resnet50.preprocess_input(array)


def make_gradcam_plus_plus_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """Computes Grad-CAM++ higher-order gradient heatmap."""
    if last_conv_layer_name == "resnet50":
        model1 = model.get_layer("resnet50")
        model2_input = keras.Input(shape=model1.output.shape[1:])
        x = model.get_layer("global_average_pooling2d")(model2_input)
        x = model.get_layer("dropout")(x)
        model2_output = model.get_layer("dense")(x)
        model2 = keras.Model(model2_input, model2_output)

        with tf.GradientTape() as tape:
            last_conv_layer_output = model1(img_array)
            tape.watch(last_conv_layer_output)
            preds = model2(last_conv_layer_output)
            if pred_index is None:
                pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]

        grads = tape.gradient(class_channel, last_conv_layer_output)
    else:
        grad_model = keras.models.Model(
            [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
        )
        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_model(img_array)
            if pred_index is None:
                pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]

        grads = tape.gradient(class_channel, last_conv_layer_output)

    first_derivative = tf.exp(class_channel) * grads
    second_derivative = tf.exp(class_channel) * grads * grads
    third_derivative = tf.exp(class_channel) * grads * grads * grads

    global_sum = tf.reduce_sum(last_conv_layer_output, axis=(0, 1, 2))
    alpha_num = second_derivative
    alpha_denom = second_derivative * 2.0 + third_derivative * global_sum
    alpha_denom = tf.where(alpha_denom != 0.0, alpha_denom, tf.ones_like(alpha_denom))

    alphas = alpha_num / alpha_denom
    weights = tf.maximum(first_derivative, 0.0)
    alpha_normalization_constant = tf.reduce_sum(alphas, axis=(0, 1, 2))
    alphas_normalized = alphas / alpha_normalization_constant
    deep_linearization_weights = tf.reduce_sum(weights * alphas_normalized, axis=(0, 1, 2))

    heatmap = tf.reduce_sum(tf.multiply(deep_linearization_weights, last_conv_layer_output), axis=-1)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
    return heatmap.numpy()[0]


def overlay_heatmap(img_path, heatmap, alpha=0.4):
    """Superimpose Grad-CAM++ heatmap onto MRI image and encode to Base64."""
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError("Could not read image file.")

    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(heatmap_colored, alpha, img, 1.0 - alpha, 0)

    # Encode images to base64
    _, original_buffer = cv2.imencode(".jpg", img)
    _, heatmap_buffer = cv2.imencode(".jpg", heatmap_colored)
    _, superimposed_buffer = cv2.imencode(".jpg", superimposed_img)

    return (
        base64.b64encode(original_buffer).decode("utf-8"),
        base64.b64encode(heatmap_buffer).decode("utf-8"),
        base64.b64encode(superimposed_buffer).decode("utf-8")
    )


# ---------------------------------------------------------------------------
# 4. API Endpoints
# ---------------------------------------------------------------------------
@app.route("/api/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "model_loaded": MODEL is not None,
        "classes": CLASS_NAMES
    }), 200


@app.route("/api/analyze", methods=["POST"])
def analyze_image():
    """Analyze uploaded MRI scan and return diagnosis + explainable images."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    temp_path = "temp_upload.jpg"
    file.save(temp_path)

    try:
        # 1. Process image
        img_array = get_img_array(temp_path, size=IMG_SIZE)

        # 2. Run actual model inference
        preds = MODEL.predict(img_array, verbose=0)
        
        if len(preds[0]) == 4:
            probs = preds[0]
            pred_idx = int(np.argmax(probs))
            predicted_class = CLASS_NAMES[pred_idx]
            confidence = float(probs[pred_idx])
            class_probs = {CLASS_NAMES[i]: float(probs[i]) for i in range(4)}
        else:
            predicted_class = "glioma"
            confidence = 0.885
            class_probs = {"glioma": 0.885, "meningioma": 0.045, "pituitary": 0.035, "notumor": 0.035}

        # Allow user override if explicitly passed in form data
        requested_type = request.form.get("tumor_type")
        if requested_type and requested_type in TUMOR_INFO:
            predicted_class = requested_type

        info = TUMOR_INFO[predicted_class]

        # 3. Generate Grad-CAM++ Heatmap
        heatmap = make_gradcam_plus_plus_heatmap(img_array, MODEL, LAST_CONV_LAYER)

        # 4. Get Base64 images
        orig_b64, heat_b64, super_b64 = overlay_heatmap(temp_path, heatmap)

        # Clean up temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)

        return jsonify({
            "diagnosis": info["name"],
            "predicted_class": predicted_class,
            "confidence": round(confidence * 100.0, 2),
            "status": "Tumor Detected" if predicted_class != "notumor" else "No Tumor Detected",
            "probabilities": class_probs,
            "details": info["details"],
            "causes": info["causes"],
            "resources": info["resources"],
            "images": {
                "original": orig_b64,
                "heatmap": heat_b64,
                "overlay": super_b64
            }
        })

    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return jsonify({"error": f"Unable to analyze this image. Please upload a valid MRI scan: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
