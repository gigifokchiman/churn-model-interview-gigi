# Interview work summary

1. Assumptions 
- In production, usually we have a template to start with (with pipelines, code templates, etc.)
- As the things were built from scratch for this interview, the work order may slightly deviate from the best practices. 

2. Initial settings
- Setting up branches, including dev, main and features/**

3. Initial deliveries from the data scientists
- Their scripts were located in the folder "notebooks" and were pushed to the features/mlops-xgboost branch.
- The mlops engineer can get the files from the dev branch with a push requests. 

4. MLOps - local development
- The files were pulled from dev branch.
- The python version was not provided. This seems to be python3.10 but it can be communicated with users in reality.
- As I was using macbook for local development and xgboost cannot support GPU in macOS, I have developed 2 requirement files 
  - requirements-train.txt for non-GPU setting and 
  - requirements-train-gpu.txt for GPU setting.
- I used both ```pip install``` and ```brew``` for the settings. For reference only, you may run
  - ```bash scripts/setup/setup_mac.sh```
- The code can be run smoothly under the newly created virtual environment. ```python notebooks/xgboost.py```
- I run the script twice with the same result. Likely that the randomness can be controlled properly. 
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

