# Customer Churn Prediction Project

## Overview

This project implements a machine learning pipeline for predicting customer churn. The pipeline includes data
processing, model training, experiment tracking, inference, and monitoring capabilities. Data Scientists can explore the
mlops work by pushing to the `dev` branch. All processes are Containerized with Docker and automated using GitHub
Actions.

## Project Structure

```
├── src/
│   ├── data/        # Data loading and preprocessing
│   ├── model/       # Model training and evaluation
│   ├── monitoring/  # Model monitoring and reporting
│   ├── pipelines/   # Main pipeline scripts with the config file
│   └── utils/       # Helper utilities
├── infra/          # Settings enables local testing using docker compose
├── tests/          # Unit tests
├── notebooks/      # Jupyter notebooks for exploration
├── .github/workflows/  # GitHub Actions workflows
└── docker/         # Dockerfile and compose files
```

## Environment Setup

### Prerequisites

- Docker https://docs.docker.com/desktop/setup/install/mac-install/
- Python 3.10 or virtual environments https://www.freecodecamp.org/news/how-to-setup-virtual-environments-in-python/

### Clone the repository

```bash
git clone https://github.com/gigifokchiman/churn-model-interview-gigi.git
cd churn-model-interview-gigi
```

### Spin up infrastructure

To set up MLflow and MinIO (local testing of AWS S3)

```bash
cd infra
docker compose -f docker-compose.yml up -d
```

## Core pipelines

The core pipeline logic can be found under `src/pipelines`.

### Build the Docker image

```bash
docker build -t churn-model-train -f Dockerfile.train .
docker build -t churn-model-inference -f Dockerfile.inference .
```

### Data generation

- To generate the sample data to MinIO:

```bash
docker run --network host churn-model-train python -m src.pipelines.data_pipeline
```

### Model training

- To run the training pipeline:

```bash
docker run --network host churn-model-train
```

- Automating the training pipeline upon model or training data changes can be done in these ways:

### Inferencing

To run the inference pipeline:

```bash
docker run --network host \
 --env-file src/pipelines/.env.test \
 -e ARTIFACTS_DIR="1/7b57243eebb14207ab25ab3190dbb5c5/artifacts" \ 
 churn-model-inference
```

`ARTIFACTS_DIR` is the directory where the model artifacts are stored in MinIO.

## Additional notes

### About XGBoost

If you encounter `libomp` related errors, refer the installation script at `scripts/setup/setup_mac.sh`. Mac do not
support GPU-accelerated XGBoost, we should use the CPU version instead.

### Scaling

Spark for data pipeline

## Appendix

### 1. Assumptions

- As the pipelines were built from scratch for this assignment, it may not fully reflect production-level best
  practices.
- All tests are run locally instead of on AWS to avoid unnecessary costs on cloud.

### 2. Branches

- Setting up branches, including `dev`, `main` and `features/**`.

### 3. Initial deliverables from the data scientists.

- Their scripts were located in the folder `notebooks` and were pushed to the `features/mlops-xgboost` branch.
- The MLOps engineer can get the files from the `dev` branch with a push requests.

### 4. Code refactoring

- Refactoring can have various level of maturity. For this exercise, we aim to separate the training and inferencing
  pipeline only.
- Minimal requirements for inferencing pipeline
    - The training part is removed.
    - The metrics part is tricky - it depends if the "ground truth" is available in production when the batch job 1
    - The data transformation parameters should be stored and used for data to be used in inferencing instead of
      deriving from the data. The parameters should be updated in retraining.
- Suggestion to the original training script: it is better to derive the median for data imputation from the TRAINING
  data only.

### 5. Set a docker compose for local testing

- Simple script for local testing in the folder `infra/docker-compose-postgres.yml`

### 6. MLOps - local development

- The python version was not provided. This seems to be python3.10 but it can be communicated with users in reality.
- As I was using macbook for local development and xgboost cannot support GPU in macOS, I have developed 2 requirement
  files
    - requirements-train.txt for non-GPU setting and
    - requirements-train-gpu.txt for GPU setting.
    - reference: https://xgboost.readthedocs.io/en/latest/install.html
- I used both ```pip install``` and ```brew``` for the settings. For reference only, you may run
    - ```bash scripts/setup/setup_mac.sh```
- The code can be run smoothly under the newly created virtual environment. ```python notebooks/xgboost.py```
- I ran the script twice with the same result. Likely that the randomness can be controlled properly.
- In reality, some sample result can be used to verify via mlflow tracking or manual sharing from data scientist.

```text
ic| train_metrics: {'Accuracy': 0.8612857142857143,
                    'F1 Score': 0.8615165200855717,
                    'Precision': 0.86188528488538,
                    'ROC AUC Score': 0.861286003287307,
                    'Recall': 0.8611480707089907}
ic| test_metrics: {'Accuracy': 0.49566666666666664,
                   'F1 Score': 0.5014827018121911,
                   'Precision': 0.4922380336351876,
                   'ROC AUC Score': 0.4957788841088787,
                   'Recall': 0.5110812625923439}
```
