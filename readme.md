# Customer Churn Prediction Project

## Overview
This project implements a machine learning pipeline for predicting customer churn. 
It includes data processing, model training, inference, and monitoring capabilities, all containerized with Docker and automated using GitHub Actions.
Data Scientists can explore the mlops work by pushing to the dev branch. 

Key features include mlflow experiment tracking and docker for inferencing.


## Project Structure
├── src/
│ ├── data/ # Data loading and preprocessing
│ ├── model/ # Model training and evaluation
│ ├── monitoring/ # Model monitoring and reporting
│ ├── pipelines/ # Main pipeline scripts with the config file there
│ └── utils/ # Helper utilities
├── infra/ # settings enables local testing using docker compose
├── tests/ # Unit tests
├── notebooks/ # Jupyter notebooks for exploration
├── .github/workflows/ # GitHub Actions workflows
└── docker/ # Dockerfile and compose files

## Environment setup
### Prerequisites
- Docker https://docs.docker.com/desktop/setup/install/mac-install/
- Python 3.10 or virtual environments https://www.freecodecamp.org/news/how-to-setup-virtual-environments-in-python/

### Setup in mac
1. Clone the repository:
```bash
git clone https://github.com/gigifokchiman/churn-model-interview-gigi.git
cd churn-prediction
```

2. Configure environment variables:
Create a .env.test file in src/pipelines. The configurations below are for testing purposes.
```
MLFLOW_S3_ENDPOINT_URL=http://localhost:9000
AWS_ACCESS_KEY_ID=minio
AWS_SECRET_ACCESS_KEY=minio123
MLFLOW_S3_IGNORE_TLS=true
MLFLOW_TRACKING_URI=http://localhost:15000
```

To set up the mlflow tracking and minIO (local testing of AWS S3)
```bash
cd infra
docker compose -f docker-compose.yml up -d
```

3. Additional notes for xgboost settings
If you enter error related to "libomp", you may refer to scripts/setsup/setup_mac.sh
Also, GPU version of xgboost is not supported in macbook, please use the non-GPU version.

4. Explore pipelines
Key pipelines can be found in src/pipelines. You may also find the 

### Scaling
- Spark for data pipeline

## Appendix - very detailed discussion
1. Assumptions
- As the things were built from scratch for this interview, the work order may slightly deviate from the best practices. 
- Local testing is preferred for this exercise to avoid AWS spending. 

2. Branches
- Setting up branches, including dev, main and features/**.

3. Initial deliveries from the data scientists.
- Their scripts were located in the folder "notebooks" and were pushed to the features/mlops-xgboost branch.
- The mlops engineer can get the files from the dev branch with a push requests. 

4. Code refactoring
- Refactoring can have various level of maturity. For this exercise, we aim to separate the training and inferencing pipeline only. 
- Minimal requirements for inferencing pipeline
  - The training part is removed. 
  - The metrics part is tricky - it depends if the "ground truth" is available in production when the batch job 1
  - The data transformation parameters should be stored and used for data to be used in inferencing instead of deriving from the data. The parameters should be updated in retraining.
- Suggestion to the original training script: it is better to derive the median for data imputation from the TRAINING data only. 

5. Set a docker compose for local testing
- Simple script for local testing in the folder "infra/docker-compose-postgres.yml"

6. MLOps - local development
- The python version was not provided. This seems to be python3.10 but it can be communicated with users in reality.
- As I was using macbook for local development and xgboost cannot support GPU in macOS, I have developed 2 requirement files 
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
