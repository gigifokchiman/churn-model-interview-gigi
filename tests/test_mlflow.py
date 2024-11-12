import unittest
import mlflow
import pandas as pd
import numpy as np
import os
import tempfile
import random
import string
import shutil
from mlflow.tracking import MlflowClient
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pytest

# Common environment setup (for local testing only).
os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'http://localhost:9000'
os.environ['AWS_ACCESS_KEY_ID'] = 'minio'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'minio123'
os.environ['MLFLOW_S3_IGNORE_TLS'] = 'true'


def generate_random_suffix(length=6):
    """Generate random string suffix."""
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))


@pytest.fixture(scope="session")
def mlflow_client():
    return MlflowClient("http://localhost:15000")


@pytest.fixture(scope="session")
def experiment_setup(mlflow_client):
    random_suffix = generate_random_suffix()
    experiment_name = f"test_experiment_{random_suffix}"
    mlflow_client.create_experiment(experiment_name)
    return experiment_name


class TestMLflowTracking(unittest.TestCase):
    def setUp(self):
        """Set up test environment before each test."""
        self.mlruns_dir = tempfile.mkdtemp()
        self.tracking_uri = "http://localhost:15000"
        random_suffix = generate_random_suffix()
        self.experiment_name = f"test_experiment_{random_suffix}"

        mlflow.set_tracking_uri(self.tracking_uri)
        self.client = MlflowClient()

        existing_experiment = mlflow.get_experiment_by_name(self.experiment_name)
        if existing_experiment:
            if existing_experiment.lifecycle_stage == 'deleted':
                self.client.restore_experiment(existing_experiment.experiment_id)
            self.experiment_id = existing_experiment.experiment_id
        else:
            self.experiment_id = mlflow.create_experiment(self.experiment_name)

        mlflow.set_experiment(self.experiment_name)

        # Create test data.
        np.random.seed(42)
        self.test_data = pd.DataFrame({
            'feature1': np.random.rand(100),
            'feature2': np.random.rand(100),
            'target': np.random.randint(0, 2, 100)
        })

    def get_run_id(self, run):
        """Helper method to get run ID."""
        return run.info.run_id if hasattr(run, 'info') else run.run_id

    def test_basic_logging(self):
        """Test basic parameter and metric logging."""
        with mlflow.start_run(experiment_id=self.experiment_id) as run:
            run_id = self.get_run_id(run)

            mlflow.log_param("learning_rate", 0.01)
            mlflow.log_param("batch_size", 32)
            mlflow.log_metric("accuracy", 0.85)
            mlflow.log_metric("loss", 0.15)

            run_data = self.client.get_run(run_id)
            self.assertEqual(run_data.data.params["learning_rate"], "0.01")
            self.assertEqual(run_data.data.metrics["accuracy"], 0.85)

    def test_artifact_logging(self):
        """Test artifact logging functionality."""
        with mlflow.start_run(experiment_id=self.experiment_id) as run:
            run_id = self.get_run_id(run)

            # Test text artifact.
            with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
                f.write("Test artifact content")
                test_file_path = f.name
            mlflow.log_artifact(test_file_path, "test_artifacts")
            os.unlink(test_file_path)

            # Test CSV artifact.
            with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
                self.test_data.to_csv(f.name, index=False)
                csv_path = f.name
            mlflow.log_artifact(csv_path, "data")
            os.unlink(csv_path)

            # Test plot artifact.
            plt.figure()
            plt.scatter(self.test_data['feature1'], self.test_data['feature2'])
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                plt.savefig(f.name)
                plot_path = f.name
            plt.close()
            mlflow.log_artifact(plot_path, "figures")
            os.unlink(plot_path)

            artifacts = self.client.list_artifacts(run_id)
            artifact_paths = [artifact.path for artifact in artifacts]
            self.assertTrue(any("test_artifacts" in path for path in artifact_paths))
            self.assertTrue(any("data" in path for path in artifact_paths))
            self.assertTrue(any("figures" in path for path in artifact_paths))

    def test_tags_logging(self):
        """Test tags logging functionality."""
        with mlflow.start_run(experiment_id=self.experiment_id) as run:
            run_id = self.get_run_id(run)

            tags = {
                "developer": "test_user",
                "version": "v1.0.0",
                "dataset": "test_dataset",
                "experiment_suffix": self.experiment_name.split('_')[-1]
            }
            mlflow.set_tags(tags)

            run_data = self.client.get_run(run_id)
            for key, value in tags.items():
                self.assertEqual(run_data.data.tags[key], value)

    def test_model_logging(self):
        """Test model logging functionality."""
        with mlflow.start_run(experiment_id=self.experiment_id) as run:
            run_id = self.get_run_id(run)

            X = self.test_data[['feature1', 'feature2']]
            y = self.test_data['target']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            model = LogisticRegression()
            model.fit(X_train, y_train)

            mlflow.sklearn.log_model(model, "model")

            signature = mlflow.models.infer_signature(
                X_train,
                model.predict(X_train)
            )
            mlflow.sklearn.log_model(
                model,
                "model_with_signature",
                signature=signature
            )

            artifacts = self.client.list_artifacts(run_id)
            artifact_paths = [artifact.path for artifact in artifacts]
            self.assertTrue(any("model" in path for path in artifact_paths))

    def tearDown(self):
        """Clean up after tests."""
        try:
            mlflow.delete_experiment(self.experiment_id)
            shutil.rmtree(self.mlruns_dir, ignore_errors=True)
        except Exception as e:
            print(f"Error during cleanup: {str(e)}")


if __name__ == '__main__':
    unittest.main(verbosity=2)
