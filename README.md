# DocLayOut - Document Layout Detection MLOps

End-to-end MLOps pipeline for fine-tuning YOLO11 models on document layout detection using Ray + MLflow on AWS EKS.

## Overview

- **Dataset**: [DocLayNet-v1.2](https://huggingface.co/datasets/ds4sd/DocLayNet-v1.2)
- **Model**: [Ultralytics YOLO11](https://huggingface.co/Ultralytics/YOLO11)
- **Infrastructure**: AWS EKS with GPU nodes
- **Training**: Ray distributed training
- **Tracking**: MLflow for experiments and model registry
- **Storage**: S3 for artifacts and models

## Project Structure

```
DocLayOut/
├── README.md
├── src/
│   ├── infra/           # Infra provisioning
│   ├── training/        # Training pipeline
│   │   ├── train.py
│   │   ├── Dockerfile
│   │   ├── requirements.txt
│   │   └── k8s/
│   │       ├── job.yaml
│   │       └── mlflow.yaml
│   ├── deployment/      # Model serving
│   │   ├── serve.py
│   │   ├── Dockerfile
│   │   └── k8s/
│   │       └── deployment.yaml
│   └── utils/            # Utility scripts
│       ├── data_prep.py
│       ├── s3_setup.py
│       └── model_utils.py
```

## Quick Start

1. **Setup S3 and permissions**
   ```bash
   python src/etc/s3_setup.py
   ```

2. **Prepare dataset**
   ```bash
   python src/etc/data_prep.py
   ```

3. **Deploy MLflow**
   ```bash
   kubectl apply -f src/training/k8s/mlflow.yaml
   ```

4. **Start training**
   ```bash
   kubectl apply -f src/training/k8s/job.yaml
   ```

## Configuration

Set these environment variables:
```bash
export AWS_REGION=us-west-2
export S3_BUCKET=doclayout-bucket
export HF_TOKEN=your_huggingface_token
export MLFLOW_TRACKING_URI=http://mlflow-service:5000
```

## S3 Bucket Structure

```
doclayout-bucket/
├── data/
│   ├── raw/
│   └── processed/
├── models/
│   ├── checkpoints/
│   └── final/
└── experiments/
    ├── logs/
    └── artifacts/
```