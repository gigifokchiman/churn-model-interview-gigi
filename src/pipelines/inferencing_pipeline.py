import os

import pandas as pd
from databricks.sdk.service.pipelines import ReportSpec
from pyarrow import timestamp
from datetime import datetime
import json
import io

from src.model.train import ChurnModelTrainer
from src.pipelines.config import get_config
from src.utils.logger import setup_logger
from src.utils.saving_data_minio import read_from_minio, save_to_minio, save_json_to_minio
from src.monitoring.model_monitor import ModelMonitor
from src.utils.saving_data_minio import save_json_to_minio


class ChurnPredictor:
    def __init__(self, artifacts_dir="/app/artifacts"):
        self.logger = setup_logger(__name__)
        self.trainer = ChurnModelTrainer(logger=self.logger)
        self.artifacts_dir = artifacts_dir
        self._load_model()

    def _load_model(self):
        try:
            model_data, _, _, self.imputation_params = \
                self.trainer.load_artifacts(self.artifacts_dir)
            self.model = model_data["model"]
            self.feature_names = model_data["feature_names"]
        except Exception as e:
            self.logger.error(f"Failed to load model: {str(e)}")
            raise

    @staticmethod
    def preprocess_data(df: pd.DataFrame, feature_names=None) -> pd.DataFrame:
        """
        One-hot encode while ensuring alignment with training features
        """
        df_encoded = pd.get_dummies(df)
        if feature_names is not None:
            df_encoded = df_encoded.reindex(columns=feature_names, fill_value=0)
        return df_encoded

    def predict(self, df: pd.DataFrame, threshold=0.5):
        try:
            # Apply preprocessing
            df_processed = self.preprocess_data(df, self.feature_names)
            df_processed = df_processed.fillna(self.imputation_params)

            # Make predictions
            probabilities = self.model.predict_proba(df_processed)
            predictions = (probabilities[:, 1] > threshold).astype(int)

            return predictions, probabilities[:, 1]

        except Exception as e:
            self.logger.error(f"Prediction failed: {str(e)}")
            raise


if __name__ == "__main__":
    # Initialize configurations and services
    predictor = ChurnPredictor(artifacts_dir=os.environ["ARTIFACTS_DIR"])
    config = get_config()

    # Load and preprocess data
    df_name = config.testing_params.get("dataset")
    df_raw = read_from_minio(df_name)

    # Make predictions
    threshold = config.inference_params.get("threshold")
    pred, probs = predictor.predict(df_raw, threshold=threshold)

    # Create results DataFrame
    result_df = pd.DataFrame({
        "account_id": df_raw["account_id"],
        "prediction": list(pred),
        "probability": list(probs)
    })

    # Save inference results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_to_minio(result_df, f"inference_result_{timestamp}.csv", "ml-result")

    # Generate and save monitoring report
    monitor = ModelMonitor()
    monitoring_report = monitor.generate_health_report(result_df)

    save_json_to_minio(monitoring_report, f"health_metrics_{timestamp}.json", "ml-result")
    print("\nModel Health Summary:")
    print(f"Total Samples: {monitoring_report['sample_metrics']['total_samples']}")
    print(f"Positive Rate: {monitoring_report['prediction_metrics']['positive_rate']:.3f}")
    print(f"Mean Probability: {monitoring_report['probability_metrics']['mean']:.3f}")
