import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import cv2
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------
# Sample image from the dataset you just downloaded
IMAGE_PATH = "../data/Testing/glioma/Te-gl_10.jpg"
IMG_SIZE = (224, 224)

# Load a pre-trained ResNet50 model
# NOTE: In a real scenario, you would load your trained model here:
# MODEL = keras.models.load_model('your_trained_model.keras')
MODEL = keras.applications.ResNet50(weights="imagenet")
LAST_CONV_LAYER = "conv5_block3_out"

# ---------------------------------------------------------
# 2. Medical Knowledge Base
# ---------------------------------------------------------
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
        "causes": "N/A",
        "resources": "Always consult with a certified radiologist or neurologist for a definitive medical diagnosis."
    }
}

# ---------------------------------------------------------
# 3. Helper Functions
# ---------------------------------------------------------
def get_img_array(img_path, size):
    img = keras.preprocessing.image.load_img(img_path, target_size=size)
    array = keras.preprocessing.image.img_to_array(img)
    array = np.expand_dims(array, axis=0)
    return keras.applications.resnet50.preprocess_input(array)

def make_gradcam_plus_plus_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
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
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()[0]

def overlay_heatmap(img_path, heatmap, alpha=0.4):
    img = cv2.imread(img_path)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(heatmap_colored, alpha, img, 1 - alpha, 0)
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    heatmap_colored_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    superimposed_img_rgb = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)
    
    return img_rgb, heatmap_colored_rgb, superimposed_img_rgb

# ---------------------------------------------------------
# 4. Main Execution & Visualization
# ---------------------------------------------------------
if __name__ == '__main__':
    print("Loading image...")
    img_array = get_img_array(IMAGE_PATH, size=IMG_SIZE)
    
    # In a real scenario, you'd get the predicted class dynamically from the model:
    # preds = MODEL.predict(img_array)
    # class_idx = np.argmax(preds[0])
    
    # For capstone demonstration with the 'glioma' test image:
    predicted_class = "glioma" 
    info = TUMOR_INFO[predicted_class]
    
    print(f"\n--- DIAGNOSIS: {info['name'].upper()} ---")
    print(f"Details: {info['details']}")
    print(f"Causes:  {info['causes']}")
    print(f"Helpful Links: {info['resources']}\n")
    
    print("Generating Grad-CAM++ heatmap...")
    heatmap = make_gradcam_plus_plus_heatmap(img_array, MODEL, LAST_CONV_LAYER)
    
    print("Creating visualization...")
    original_img, heatmap_img, superimposed_img = overlay_heatmap(IMAGE_PATH, heatmap)
    
    # Plotting the Explainable Visualization with text
    fig = plt.figure(figsize=(15, 8))
    
    plt.subplot(1, 3, 1)
    plt.title("Original MRI Scan")
    plt.imshow(original_img)
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.title("Grad-CAM++ Heatmap")
    plt.imshow(heatmap_img)
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.title("Explainable Visualization (Overlay)")
    plt.imshow(superimposed_img)
    plt.axis('off')
    
    # Add informational text at the bottom of the figure
    info_text = (
        f"DIAGNOSIS: {info['name'].upper()}\n\n"
        f"DETAILS: {info['details']}\n\n"
        f"CAUSES: {info['causes']}\n\n"
        f"RESOURCES / HELP: {info['resources']}"
    )
    
    plt.figtext(0.5, 0.05, info_text, wrap=True, horizontalalignment='center', 
                fontsize=12, bbox=dict(facecolor='lightgrey', alpha=0.5, pad=10))
    
    plt.subplots_adjust(bottom=0.25) # Make room for the text
    plt.savefig('explainable_visualization_result.png')
    plt.show()
    
    print("Done! Check 'explainable_visualization_result.png' for the output.")
