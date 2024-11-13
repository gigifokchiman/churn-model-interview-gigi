import random
import string

import mlflow
from mlflow.tracking import MlflowClient


def generate_random_suffix(length=6):
    """Generate random string suffix."""
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))


def get_or_create_experiment(experiment_name: str) -> str:
    """Get or create an MLflow experiment."""
    client = MlflowClient()
    experiment = mlflow.get_experiment_by_name(experiment_name)

    if experiment is None:
        # Create new experiment if it doesn't exist
        experiment_id = mlflow.create_experiment(experiment_name)
    else:
        # If experiment exists and is deleted, restore it
        if experiment.lifecycle_stage == "deleted":
            client.restore_experiment(experiment.experiment_id)
        experiment_id = experiment.experiment_id

    return experiment_id
