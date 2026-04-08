<div align="center">

# Lung and Colon Cancer Detection with Explainable AI

**A dual-stream histopathology classification project using ConvNeXtV2, Phikon, model fusion, and explainable AI**

![Python](https://img.shields.io/badge/Python-Deep%20Learning-blue)
![Task](https://img.shields.io/badge/Task-Histopathology%20Classification-darkgreen)
![Models](https://img.shields.io/badge/Models-ConvNeXtV2%20%2B%20Phikon-orange)
![Best Accuracy](https://img.shields.io/badge/Best%20Accuracy-99.08%25-crimson)

</div>

An end-to-end deep learning project for **histopathology image classification** across **five tissue classes** using a **dual-stream pipeline**:

- **CNN stream:** `ConvNeXtV2`
- **Vision Transformer stream:** `Phikon`
- **Fusion stage:** soft voting and stacked ensemble
- **Explainability stage:** Grad-CAM, Integrated Gradients, and GradientShap

The goal is to classify **lung and colon tissue images** into clinically relevant categories while keeping the final system interpretable through explainable AI methods.

---

## Project Overview

This project studies **lung and colon cancer detection from histopathology images** using two complementary deep learning models:

- A **ConvNeXtV2-based CNN** to capture strong local and hierarchical visual features.
- A **Phikon Vision Transformer (ViT)** to leverage transformer-based global representation learning, especially useful for pathology imagery.

After training the two models separately, their predictions are combined in a **fusion framework** to improve final classification performance. The project also includes an **XAI layer** to visualize which image regions influenced the model’s decision.

---

## Environment Setup

This project can be run on **Python 3.14**, but you need current package versions with prebuilt wheels for Apple Silicon.

### Recommended setup

```bash
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

If `pip` still tries to build old packages from source, make sure you are using the updated `requirements.txt` from this repository and not a cached older copy.

---

## Dataset

The project uses the **LC25000 histopathology dataset** in an augmented folder structure: `Augmented_LC25000_train/`.

### Classes used

- `colon_aca` - Colon adenocarcinoma
- `colon_n` - Benign colon tissue
- `lung_aca` - Lung adenocarcinoma
- `lung_n` - Benign lung tissue
- `lung_scc` - Lung squamous cell carcinoma

### Dataset structure

- Balanced class folders with roughly **36,000 images per class**
- Around **180,000 images** in total in the augmented training directory
- Images are resized to **224 x 224**
- Dataset split used in the notebooks:
  - **80% training**
  - **10% validation**
  - **10% testing**

### Preprocessing

The notebooks apply standard image preprocessing before training:

- resize to `224 x 224`
- convert to tensor
- normalize with ImageNet mean and standard deviation

---

## Models Used

### 1. Stream A: ConvNeXtV2

The first branch uses **`facebook/convnextv2-base-22k-224`** as a pretrained backbone.

### Why it is used

ConvNeXtV2 is a modern CNN architecture that keeps the strengths of convolutional networks while improving feature quality and transfer learning performance. It is well-suited for extracting **fine-grained local texture patterns**, which are important in histopathology images.

### Architecture used in this project

- pretrained ConvNeXtV2 backbone
- frozen feature extractor
- custom classification head:
  - `Linear(1024 -> 256)`
  - `GELU`
  - `Dropout(0.5)`
  - `Linear(256 -> 5)`

### Training notes

- loss: `CrossEntropyLoss(label_smoothing=0.1)`
- optimizer: `Adam`
- learning rate: `0.001`
- weight decay: `1e-4`
- scheduler: `ReduceLROnPlateau`

### Final ConvNeXtV2 result

- **Test Accuracy:** `96.48%`

Class-wise highlights from the saved evaluation:

- very strong performance on `colon_aca`, `colon_n`, and `lung_n`
- slightly lower performance on `lung_aca` and `lung_scc`, which are more visually challenging classes

---

### 2. Stream B: Phikon Vision Transformer

The second branch uses **`owkin/phikon`**, a pathology-focused Vision Transformer.

### Why it is used

Phikon is designed for computational pathology and is especially useful for learning **global contextual relationships** in histology images. This complements the CNN stream, which focuses more strongly on local visual structures.

### Architecture used in this project

- pretrained `Phikon` backbone
- frozen transformer base
- custom classification head:
  - `Linear(768 -> 256)`
  - `GELU`
  - `Dropout(0.5)`
  - `Linear(256 -> 5)`

### Training notes

- loss: `CrossEntropyLoss(label_smoothing=0.1)`
- optimizer: `Adam`
- learning rate: `0.001`
- weight decay: `1e-4`
- scheduler: `ReduceLROnPlateau`
- trained for **30 epochs**

### Final Phikon result

- **Final Test Loss:** `0.4276`
- **Final Test Accuracy:** `98.67%`

This model clearly outperformed the standalone CNN stream and acted as the stronger expert in the fusion pipeline.

---

## Fusion Strategy

The project does not stop at single-model classification. It combines both streams to build a more reliable diagnostic system.

### 1. Soft Voting Ensemble

In the first fusion approach, the probability outputs from ConvNeXtV2 and Phikon are combined through **soft voting**.

### Result

- **Ultimate Ensemble Accuracy:** `98.68%`

This already improves on the CNN branch and gives performance comparable to the stronger transformer model.

### 2. Stacked Ensemble

The second and best-performing fusion approach uses **stacking**:

- probabilities from ConvNeXtV2: `5` features
- probabilities from Phikon: `5` features
- combined into a `10`-dimensional meta-feature vector
- a **Logistic Regression meta-model** learns how to make the final decision

### Final fused result

- **Ultimate Stacked Ensemble Accuracy:** `99.08%`

This is the best result achieved in the project.

---

## Results Summary

| Model | Accuracy |
| --- | ---: |
| ConvNeXtV2 (CNN stream) | **96.48%** |
| Phikon (ViT stream) | **98.67%** |
| Soft Voting Fusion | **98.68%** |
| Stacked Fusion | **99.08%** |

### Interpretation

- The **CNN branch** provides strong baseline performance.
- The **Vision Transformer branch** performs significantly better on its own.
- The **stacked fusion model** gives the best result, showing that the two streams learn complementary information.
- Fusion is especially valuable because it combines **local texture understanding** from the CNN with **global context modeling** from the transformer.

---

## Explainable AI Component

One of the important parts of this project is the **explainability pipeline** added after fusion analysis.

The notebook includes a **triple-forensic visualization suite** for model interpretation using:

- **Grad-CAM**
- **Integrated Gradients**
- **GradientShap**

### What this explainability step does

For a selected histopathology image, the XAI pipeline highlights the regions that most influenced the prediction. This helps:

- understand why the model predicted a class
- verify whether attention is focused on relevant tissue structures
- improve trust in the system for medical-image applications

### Methods in short

- **Grad-CAM:** produces heatmaps from deep feature maps to show where the model is focusing
- **Integrated Gradients:** attributes the prediction to input pixels by accumulating gradients along a path from a baseline image
- **GradientShap:** estimates feature importance using gradient-based SHAP-style attribution

Together, these methods provide both **region-level** and **pixel-level** explanations.

---

## Notebook Workflow

The project is organized into three notebooks:

### `dl_cnn_model_training.ipynb`

- data loading and preprocessing
- ConvNeXtV2 model definition
- CNN training and checkpoint evaluation
- final classification report and confusion matrix

### `dl_visonT_model_training.ipynb`

- Phikon data pipeline
- ViT model training
- validation tracking across epochs
- final evaluation on the test set

### `dl_fusion.ipynb`

- loading trained ConvNeXtV2 and Phikon models
- soft voting ensemble
- stacked ensemble with logistic regression
- random sample testing
- explainable AI analysis using Captum

---

## Docker Deployment

The repository now includes a full Docker setup for the **Cancer Hospital Management App**, including the fused pathology model and observability stack.

### What runs in Docker

- the Flask app on `http://localhost:7860`
- Prometheus on `http://localhost:9090`
- Grafana on `http://localhost:3000`
- cAdvisor on `http://localhost:8080`

The Docker image pre-downloads the Hugging Face base backbones during build, so the container does not need to fetch them at runtime.

### Start the full stack

```bash
docker compose up --build
```

### Grafana login

- username: `admin`
- password: `admin`

### Metrics included

- per-route HTTP hits
- per-route request latency
- inference counts
- prediction and Grad-CAM latency
- model artifact readiness
- app process CPU and memory
- container CPU and memory through cAdvisor

### Optional GPU monitoring

If you are on an **NVIDIA Linux Docker host**, you can also start the DCGM exporter:

```bash
docker compose --profile gpu up --build
```

Important:

- GPU monitoring inside Docker is **not available on Apple Silicon macOS containers**
- on macOS you will still get app metrics, route hits, CPU, RAM, and container monitoring
- the optional GPU profile is meant for NVIDIA-enabled Docker environments

---

## Key Takeaways

- This project demonstrates a **hybrid pathology AI system** for **lung and colon cancer classification**.
- A **CNN + Vision Transformer** combination is more effective than relying on a single model alone.
- The **stacked fusion model reached 99.08% accuracy**, the best performance in the project.
- The added **explainable AI pipeline** makes the system more transparent and suitable for medical AI research settings.

---

## Future Improvements

- train both backbones with partial unfreezing for deeper fine-tuning
- use patient-level or slide-level validation if available
- add external validation on another histopathology dataset
- extend explainability to the transformer branch as well
- package the fused model into a deployable web app or clinical decision-support prototype

---

## Files in This Repository

- `dl_cnn_model_training.ipynb` - CNN stream training and evaluation
- `dl_visonT_model_training.ipynb` - Vision Transformer training and evaluation
- `dl_fusion.ipynb` - ensemble learning and explainability
- `convnext_v2_epoch_30.pth` - saved ConvNeXtV2 checkpoint
- `phikon_best_overall.pth` - saved Phikon checkpoint
- `ultimate_patho_fusion_v1.pkl` - saved stacked fusion model generated in the fusion workflow

---

## Conclusion

This project presents a strong deep learning framework for **automated lung and colon cancer histopathology classification**. By combining **ConvNeXtV2**, **Phikon**, **stacked fusion**, and **explainable AI**, the system achieves both **high predictive performance** and **better interpretability**, which are both critical in medical imaging applications.
