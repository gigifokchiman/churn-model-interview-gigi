import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, \
    roc_auc_score


def calculate_metrics(y_true, y_pred, y_prob=None):
    metrics = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred)),
        'recall': float(recall_score(y_true, y_pred)),
        'f1': float(f1_score(y_true, y_pred))
    }

    if y_prob is not None:
        metrics['roc_auc'] = float(roc_auc_score(y_true, y_prob))

    return metrics


def calculate_drift_metrics(reference_data, current_data):
    """Calculate distribution drift between training and current data"""
    drift_metrics = {}

    for column in reference_data.columns:
        if reference_data[column].dtype in ['int64', 'float64']:
            drift_metrics[column] = {
                'mean_diff': float(
                    current_data[column].mean() - reference_data[column].mean()),
                'std_diff': float(
                    current_data[column].std() - reference_data[column].std()),
                'ks_statistic': float(np.abs(
                    current_data[column].kurt() - reference_data[column].kurt()))
            }

    return drift_metrics
