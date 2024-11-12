import os
from datetime import datetime
from typing import Dict, Tuple

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost.sklearn import XGBClassifier

from src.utils.logger import setup_logger


class ChurnModelTrainer:
    def __init__(self, config: Dict = None, logger=None):
        self.config = config or {}
        self.model = None
        self.logger = logger or setup_logger(__name__)
        self.training_columns = None
        self.label_column = None
        self.imputation_params = None

    def prepare_data(self, df: pd.DataFrame) -> Tuple:
        """
        Prepare data for training.
        """
        omit_columns = ["account_id"]
        str_columns = ["industry_category", "operating_country", "revenue_trend"]
        label_column = "is_churned"

        df = pd.get_dummies(df, columns=str_columns)
        self.training_columns = [
            str(i) for i in df.columns if i not in {label_column, *omit_columns}
        ]

        self.label_column = label_column

        # TODO: fix the imputation_params to include the training data median only
        self.imputation_params = df[self.training_columns].median()

        df[self.training_columns] = df[self.training_columns].fillna(self.imputation_params)
        train_df, test_df = train_test_split(df, test_size=0.3, random_state=self.config.get('random_state', 42))
        return (
            train_df[self.training_columns],
            train_df[self.label_column],
            train_df[omit_columns],
            test_df[self.training_columns],
            test_df[self.label_column],
            test_df[omit_columns],
        )

    def train(self, X_train: pd.DataFrame, y_train: pd.Series, **params) -> None:
        """Train the model"""
        self.logger.info("Training model...")

        self.model = XGBClassifier(
            **params
        )
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series, threshold=0.5, prefix="training") -> Dict:
        """Evaluate the model"""
        self.logger.info("Evaluating model...")

        y_pred = (self.model.predict_proba(X_test)[:, 1] > threshold).astype(int)

        base_metrics = {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'precision': float(precision_score(y_test, y_pred)),
            'recall': float(recall_score(y_test, y_pred)),
            'f1': float(f1_score(y_test, y_pred)),
            'roc_auc': float(roc_auc_score(y_test, y_pred)) if len(y_test.unique()) > 1 else -0.0
        }
        metrics = {f'{prefix}_{k}': v for k, v in base_metrics.items()}
        return metrics

    def save_artifacts(self, output_dir: str, training_metrics: Dict[str, float], testing_metrics: Dict[str, float],
                       imputation_params) -> \
        tuple[str, str, str, str]:
        """Save model artifacts and metrics"""
        self.logger.info("Saving artifacts...")

        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        model_path = os.path.join(output_dir, f'model.pkl')
        training_metrics_path = os.path.join(output_dir, f'training_metrics.json')
        testing_metrics_path = os.path.join(output_dir, f'testing_metrics.json')
        imputation_params_path = os.path.join(output_dir, f'inputation_params.json')

        try:
            joblib.dump({
                'model': self.model,
                'feature_names': self.training_columns,
                'config': self.config
            }, model_path)

            pd.Series(training_metrics).to_json(training_metrics_path)
            pd.Series(testing_metrics).to_json(testing_metrics_path)
            pd.Series(imputation_params).to_json(imputation_params_path)

            return model_path, training_metrics_path, testing_metrics_path, imputation_params_path

        except Exception as e:
            error_msg = f"Failed to save artifacts: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg)

    def load_artifacts(self, output_dir) -> \
        tuple[Dict, Dict[str, float], Dict[str, float], Dict]:
        """Load model artifacts and metrics"""
        self.logger.info("Loading artifacts...")

        model_path = os.path.join(output_dir, f'model.pkl')
        training_metrics_path = os.path.join(output_dir, f'training_metrics.json')
        testing_metrics_path = os.path.join(output_dir, f'testing_metrics.json')
        imputation_params_path = os.path.join(output_dir, f'inputation_params.json')
    
        try:
            # Load model and related data
            model_data = joblib.load(model_path)
            self.model = model_data['model']
            self.training_columns = model_data['feature_names']
            self.config = model_data.get('config', {})

            # Load metrics and parameters
            training_metrics = pd.read_json(training_metrics_path, typ='series').to_dict()
            testing_metrics = pd.read_json(testing_metrics_path, typ='series').to_dict()
            imputation_params = pd.read_json(imputation_params_path, typ='series').to_dict()

            self.logger.info("Successfully loaded all artifacts")
            return model_data, training_metrics, testing_metrics, imputation_params

        except FileNotFoundError as e:
            self.logger.error(f"Failed to find artifact file: {str(e)}")
            raise

        except Exception as e:
            self.logger.error(f"Error loading artifacts: {str(e)}")
            raise RuntimeError(f"Failed to load artifacts: {str(e)}")
