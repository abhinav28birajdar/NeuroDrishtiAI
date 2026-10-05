# NeuroDrishti AI

## Current project explanation

NeuroDrishti AI is a research and educational application for four-class brain
MRI image classification. The project combines a TensorFlow/Keras ResNet50
model with Grad-CAM++ visual explanations and a Streamlit dashboard.

The application is intended to demonstrate model inference, explainability,
confidence visualization, report generation, and API integration. It is not a
medical device and must not be used as a substitute for a qualified
professional's diagnosis or treatment decision.

## Current capabilities

### Streamlit dashboard

The main entry point is `streamlit_app.py`. The dashboard currently provides:

- MRI upload for JPG, JPEG, and PNG images
- 224 x 224 image preprocessing using ResNet50 preprocessing
- Four-class prediction:
  - Glioma
  - Meningioma
  - No Tumor
  - Pituitary Tumor
- Prediction, confidence, and clinical-status metric cards
- Original MRI, activation heatmap, and explainable overlay images
- Interactive Plotly confidence breakdown graph
- Confidence breakdown repeated inside the on-screen diagnostic report
- Clinical context and resource information for each predicted class
- Downloadable PDF reports generated with ReportLab
- Session-only analysis history in the sidebar
- A clear-history control for the current browser session
- Model Information and About pages
- NeuroDrishti AI favicon and sidebar logo from `assests/appicon.png`

The history is intentionally stored in `st.session_state`. It is not written
to a database or shared between users, and it disappears when the Streamlit
session ends.

### Diagnostic report

After an image is analyzed, the dashboard displays:

1. Analysis metadata
2. Primary diagnosis, confidence score, and clinical status
3. Clinical summary, risk factors, and resource links
4. Interactive multi-class confidence graph and probability table
5. Research-only clinical disclaimer

The PDF report contains the prediction summary, class probabilities,
explainability images, clinical context, and medical disclaimer. JSON is used
by the REST API response; the Streamlit dashboard does not provide a JSON
download button.

## Inference pipeline

```text
Uploaded MRI image
        |
        v
Temporary image file
        |
        v
Resize to 224 x 224 and apply ResNet50 preprocessing
        |
        v
TensorFlow/Keras model prediction
        |
        +--> Four-class probabilities
        |
        +--> Highest-probability diagnosis
        |
        v
Grad-CAM++ higher-order gradients
        |
        v
OpenCV heatmap and blended MRI overlay
        |
        v
Streamlit dashboard, history, graph, and report
```

The application first attempts to load `brain_tumor_model.keras` or
`model.keras`. If a compatible custom model is unavailable, it falls back to an
ImageNet ResNet50 model for demonstration. The fallback is not a trained
brain-tumor classifier and should not be interpreted as clinical inference.

## Explainability implementation

The Grad-CAM++ implementation calculates first-, second-, and third-order
gradient terms for the selected class. It identifies the final convolutional
feature representation, normalizes the activation map, applies OpenCV
`COLORMAP_JET`, and blends the result with the original scan using an alpha
value of approximately `0.4`.

Warm colors in the heatmap indicate regions that contributed more strongly to
the model's class score. These visualizations are model explanations, not
segmentation masks or proof of disease.

## Confidence visualization

The confidence breakdown is generated with Plotly Express as a horizontal bar
chart. The chart:

- Displays all four supported classes
- Sorts classes by confidence
- Shows percentage values on the bars
- Uses a blue continuous confidence scale
- Provides hover details
- Uses a percentage axis capped at a readable range
- Appears in both the analysis dashboard and the diagnostic report

## Application structure

```text
NeuroDrishti-AI/
├── streamlit_app.py              # Main Streamlit application
├── report_generator.py           # ReportLab PDF report generation
├── requirements.txt              # Runtime and deployment dependencies
├── runtime.txt                   # Streamlit Cloud Python runtime declaration
├── prepare_dataset.py            # Dataset validation and preparation utility
├── train.py                      # Model training and evaluation pipeline
├── PROJECT_EXPLANATION.md        # This document
├── README.md                     # Short project introduction and setup guide
├── api/
│   └── app.py                    # Flask inference API
├── assests/
│   ├── appicon.png               # Application logo and favicon
│   ├── Plots.txt                 # Plot inventory
│   ├── RestNet_plots/            # ResNet training plots
│   ├── Attention_U_Net_plots/    # Segmentation plots
│   ├── Yolo_plots_all/           # YOLO experiment plots
│   └── Multi_class_Yolo_Plots_/  # Multi-class YOLO plots
├── notebooks/                    # Training and research experiments
└── .claude/skills/               # Required project development guidance
```

The `assests` spelling is part of the current repository structure and is used
by the app for the logo path.

## REST API

The optional Flask service is implemented in `api/app.py`.

```powershell
python api/app.py
```

Available endpoints:

```text
GET  /api/health
POST /api/analyze
```

`/api/analyze` accepts an uploaded MRI image and returns JSON containing the
predicted class, confidence information, class probabilities, and
base64-encoded explainability images where available. CORS is enabled for
clients that need to call the service from another origin.

## Dataset and research utilities

`prepare_dataset.py` validates and prepares image data for model experiments.
It supports image readability checks, duplicate detection, class reporting, and
reproducible dataset organization.

`train.py` contains the TensorFlow/Keras training and test-evaluation pipeline.
The repository also includes notebooks exploring:

- ResNet50 multi-class classification
- VGG16 and VGG19 comparisons
- EfficientNet experiments
- Binary classification
- U-Net and Attention U-Net segmentation
- YOLO detection
- Multi-class YOLO detection
- Grad-CAM visualization

The segmentation and detection notebooks are experimental research components;
they are not invoked by the Streamlit dashboard at runtime.

## Runtime and deployment

The current deployment environment uses Python 3.14 as declared in
`runtime.txt`. `requirements.txt` pins `tensorflow==2.22.0rc0` because the
deployment logs identified it as the available TensorFlow build with a Python
3.14 wheel.

Install and run locally:

```powershell
pip install -r requirements.txt
python -m streamlit run streamlit_app.py
```

The dashboard is available at `http://localhost:8501`.

## Safety and limitations

- Predictions are research outputs, not diagnoses.
- Grad-CAM++ highlights model saliency, not confirmed tumor boundaries.
- A fallback ImageNet model is not suitable for brain-tumor classification.
- Results depend on model weights, preprocessing, image quality, and training
  data.
- Session history is temporary and is not an audit trail or medical record.
- Clinical decisions must be made by qualified healthcare professionals.
