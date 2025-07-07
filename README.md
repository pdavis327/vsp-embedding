# VSP Eyeglass Frame Embedding Project

A comprehensive demonstration of OpenShift AI for VSP (Vision Service Plan), showcasing frame style prediction for eyeglass frames using BGE-Large embeddings and machine learning.

## 🎯 Project Overview

This project demonstrates how to predict the frame style for eyeglass frames using:
- **BGE-Large embeddings** model served via vLLM
- **K-Nearest Neighbors (KNN)** classification
- **Kubeflow Pipelines** for ML workflow orchestration
- **OpenShift AI** infrastructure

The system analyzes frame descriptions and predicts whether frames belong to styles like Classic, Modern, Vintage, Sporty, Luxury, Fashion, Professional, or Youthful.

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Synthetic     │    │   vLLM Server   │    │   Kubeflow      │
│   Data          │───▶│   (BGE-Large)   │───▶│   Pipeline      │
│   Generation    │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │   Embeddings    │    │   Model         │
                       │   Generation    │    │   Training &    │
                       │                 │    │   Evaluation    │
                       └─────────────────┘    └─────────────────┘
```

## 📁 Project Structure

```
vsp-embedding/
├── app/                          # Application code
│   ├── data/                     # Dataset files
│   │   ├── synthetic_eyeglass_frames_1k.csv
│   │   └── unseen_eyeglass_frames.csv
│   ├── notebooks/                # Jupyter notebooks
│   │   └── vsp_demo.ipynb        # Interactive demo
│   ├── scripts/                  # Utility scripts
│   │   ├── synthetic-data.py     # Data generation
│   │   └── images/               # Container images
│   └── utils/                    # Utility modules
│       └── vllm_client.py        # vLLM client wrapper
├── K8s/                          # Kubernetes/OpenShift manifests
│   ├── auth/                     # Authentication setup
│   ├── gpu/                      # GPU operator configuration
│   ├── model-serving/            # Model serving components
│   └── rhoi/                     # Red Hat OpenShift AI setup
└── pipeline/                     # Kubeflow pipeline
    ├── vsp_eyeglass_frame_pipeline.py
    ├── vsp_eyeglass_frame_pipeline.yaml
    └── requirements.txt
```

## 🚀 Quick Start

### Prerequisites

- OpenShift cluster with Red Hat OpenShift AI (RHOAI) installed
- GPU resources available (for vLLM inference)
- MinIO or S3-compatible storage

## 📊 Data Generation

The project includes synthetic data generation for eyeglass frames:

```bash
cd app/scripts
python synthetic-data.py
```

The synthetic data includes:
- **Frame descriptions** with style-specific characteristics
- **8 frame styles**: Classic, Modern, Vintage, Sporty, Luxury, Fashion, Professional, Youthful
- **Realistic frame types**: metal frames, acetate frames, wire frames, rimless frames, etc.

## Notebook
[vsp demo notebook](app/notebooks/vsp_demo.ipynb).
- Run in a jupyterlab notebook in an Openshift AI workbench for the live demo. 
- Uses a [custom notebook image](app/scripts/images)


## 🔧 Pipeline Components
- pipeline is built using kubeflow and follows the same path as the notebook
- one additional step includes making predictions on unseen data. \

### 1. Data Preparation
- Loads synthetic eyeglass frame data from MinIO/S3
- Preprocesses frame descriptions
- Encodes frame style labels

### 2. Embedding Generation
- Connects to vLLM server running BGE-Large
- Generates 1024-dimensional embeddings for frame descriptions
- Handles batch processing for efficiency

### 3. Model Training
- Trains KNN classifier (k=3) on embeddings
- Splits data into training/testing sets
- Evaluates model performance

### 4. Model Evaluation
- Generates confusion matrix
- Calculates accuracy metrics
- Creates visualization plots

### 5. Prediction on Unseen Data
- Processes new, unseen eyeglass frames
- Generates embeddings and predictions
- Saves results to storage

## 🎯 Model Performance
The KNN classifier typically achieves:
- **Accuracy**: ~95% on test data
- **Embedding dimension**: 1024 (BGE-Large)
- **Training samples**: 800 (80% of dataset)
- **Test samples**: 200 (20% of dataset)

## 🐳 Container Images

The project includes Docker configurations for:
- **vLLM server** with BGE-Large model
- **Kubeflow pipeline components**
- **Jupyter workbench** with required dependencies

## 🔐 Security & Authentication

- **API key authentication** for vLLM endpoints
- **OpenShift authentication** via htpasswd
- **MinIO/S3 credentials** for data storage
- **SSL certificate handling** for internal deployments

## 📈 Monitoring & Observability

- **GPU monitoring** via DCGM exporter
- **Model metrics** tracking in Kubeflow
- **Performance dashboards** for inference
- **Logging** across all components

## 📄 License

This project is licensed under the terms specified in the LICENSE file.
