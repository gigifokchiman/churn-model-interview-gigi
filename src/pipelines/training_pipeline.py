import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import mlflow
import pandas as pd
from dotenv import load_dotenv

from src.model.train import ChurnModelTrainer
from src.pipelines.config import get_config
from src.utils.logger import setup_logger
from src.utils.mlflow_utils import get_or_create_experiment
from src.utils.saving_data_minio import read_from_minio

logger = setup_logger(name=__name__, level=logging.INFO)


@dataclass
class ModelArtifacts:
    """Container for model training artifacts and metrics."""
    model_path: str
    training_metrics_path: str
    testing_metrics_path: str
    training_metrics: Dict[str, float]
    testing_metrics: Dict[str, float]
    run_id: str


class ModelTrainingPipeline:
    """Pipeline for training and evaluating churn prediction models."""

    def __init__(
        self,
        dataset_name: str,
        model_params: Dict[str, Any],
        experiment_name: str = "",
        run_name: str = "",
    ):
        """
        Initialize the training pipeline.

        Args:
            dataset_name: Name of the dataset to use for training
            model_params: Model hyperparameters
            experiment_name: MLflow experiment name
            run_name: MLflow run name
        """
        self.run_name = run_name
        self.experiment_name = experiment_name
        self.dataset_name = dataset_name
        self.model_params = model_params
        self.trainer = ChurnModelTrainer(logger=logger)

        os.makedirs(run_name, exist_ok=True)
        self._setup_mlflow_experiment()

    def _setup_mlflow_experiment(self) -> None:
        """Set up MLflow experiment, creating it if it doesn't exist."""
        try:
            self.experiment_id = mlflow.create_experiment(self.experiment_name)
            logger.info(f"Created new MLflow experiment: {self.experiment_name}")
        except Exception:
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            self.experiment_id = experiment.experiment_id if experiment else None
            logger.info(f"Using existing MLflow experiment: {self.experiment_name}")

    def get_data(self, *args) -> pd.DataFrame:
        """
        Retrieve dataset from MinIO storage.

        Returns:
            DataFrame containing the dataset
        
        Raises:
            ValueError: If dataset loading fails
        """
        logger.info(f"Loading dataset: {self.dataset_name}")
        df = read_from_minio(self.dataset_name, *args)

        if df is None or df.empty:
            raise ValueError(f"Failed to load DataFrame from {self.dataset_name}")

        logger.info(f"Loaded dataset with shape: {df.shape}")
        return df

    def _log_metrics_and_params(
        self,
        train_size: int,
        test_size: int,
        training_metrics: Dict[str, float],
        testing_metrics: Dict[str, float],
        config: Any
    ) -> None:
        """Log all relevant metrics and parameters to MLflow."""
        mlflow.log_param("train_size", train_size)
        mlflow.log_param("test_size", test_size)

        for param, value in config.training_params.items():
            mlflow.log_param(param, value)

        mlflow.log_metrics(training_metrics)
        mlflow.log_metrics(testing_metrics)

    def _save_artifacts(
        self,
        model_path: str,
        training_metrics_path: str,
        testing_metrics_path: str,
        imputation_params_path: str
    ) -> None:
        """Log all artifacts to MLflow."""
        for path in [model_path, training_metrics_path, testing_metrics_path, imputation_params_path]:
            mlflow.log_artifact(path)
        mlflow.xgboost.log_model(self.trainer.model, "model")

    def run(self) -> ModelArtifacts:
        """
        Execute the training pipeline with MLflow tracking.

        Returns:
            ModelArtifacts containing paths and metrics

        Raises:
            RuntimeError: If pipeline execution fails
        """
        logger.info("Starting training pipeline execution")
        mlflow.set_experiment(self.experiment_name)

        with mlflow.start_run(run_name=self.run_name) as run:
            try:
                # Data preparation
                df = self.get_data()
                train_data = self.trainer.prepare_data(df)
                train_features, train_labels = train_data[:2]
                test_features, test_labels = train_data[3:5]

                if len(train_features) < 1 or len(test_features) < 1:
                    raise ValueError("Data preparation failed: Empty feature sets")

                # Training and evaluation
                self.trainer.train(train_features, train_labels)
                training_metrics = self.trainer.evaluate(train_features, train_labels)
                testing_metrics = self.trainer.evaluate(
                    test_features, test_labels, prefix="testing"
                )

                # Logging and artifact saving
                self._log_metrics_and_params(
                    len(train_features),
                    len(test_features),
                    training_metrics,
                    testing_metrics,
                    get_config()
                )

                artifacts = self.trainer.save_artifacts(
                    self.run_name,
                    training_metrics,
                    testing_metrics,
                    self.trainer.imputation_params
                )

                self._save_artifacts(*artifacts)

                return ModelArtifacts(
                    model_path=artifacts[0],
                    training_metrics_path=artifacts[1],
                    testing_metrics_path=artifacts[2],
                    training_metrics=training_metrics,
                    testing_metrics=testing_metrics,
                    run_id=run.info.run_id
                )

            except Exception as e:
                logger.error(f"Pipeline execution failed: {str(e)}", exc_info=True)
                mlflow.log_param("error", str(e))
                raise RuntimeError(f"Pipeline execution failed: {str(e)}")


def setup_mlflow_environment(env_path: Path) -> str:
    """Set up MLflow environment variables from .env file."""
    load_dotenv(env_path)

    env_vars = {
        'MLFLOW_S3_ENDPOINT_URL': os.getenv('MLFLOW_S3_ENDPOINT_URL'),
        'AWS_ACCESS_KEY_ID': os.getenv('AWS_ACCESS_KEY_ID'),
        'AWS_SECRET_ACCESS_KEY': os.getenv('AWS_SECRET_ACCESS_KEY'),
        'MLFLOW_S3_IGNORE_TLS': os.getenv('MLFLOW_S3_IGNORE_TLS', 'true')
    }

    for key, value in env_vars.items():
        if value:
            os.environ[key] = value

    return os.getenv('MLFLOW_TRACKING_URI', '')


if __name__ == "__main__":
    try:
        logger.info("Starting main execution")
        config = get_config()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        experiment_name = config.model_params.get("experiment_name", "")
        run_name = config.model_params.get("run_name", f"training_run_{timestamp}")

        if run_name:
            run_name = f"{run_name}_{timestamp}"

        # Setup MLflow environment
        env_path = Path(__file__).parent / '.env.test'
        tracking_uri = setup_mlflow_environment(env_path)

        # Initialize and run pipeline
        experiment_id = get_or_create_experiment(experiment_name)
        mlflow.set_experiment(experiment_name)

        pipeline = ModelTrainingPipeline(
            dataset_name=config.training_params.get("dataset"),
            model_params=config.training_params.get("model"),
            experiment_name=experiment_name,
            run_name=run_name
        )

        artifacts = pipeline.run()

        # Log results
        logger.info(f"Model saved to: {artifacts.model_path}")
        logger.info(f"Metrics saved to: {artifacts.training_metrics_path}")
        logger.info(f"Model performance: {artifacts.training_metrics}")

        mlflow_ui_url = (
            f"{mlflow.get_tracking_uri()}/#/experiments/"
            f"{mlflow.get_experiment_by_name(experiment_name).experiment_id}"
            f"/runs/{artifacts.run_id}"
        )
        logger.info(f"MLflow UI: {mlflow_ui_url}")

    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        raise
