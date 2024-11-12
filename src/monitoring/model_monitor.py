# src/monitoring/model_monitor.py
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from typing import Dict

class ModelMonitor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def generate_health_report(self, predictions: pd.DataFrame) -> Dict:
        """
        Generate basic health metrics report
        
        Args:
            predictions (pd.DataFrame): DataFrame with columns 
                ['account_id', 'prediction', 'probability']
        
        Returns:
            Dict: Basic health metrics
        """
        try:
            report = {
                'timestamp': datetime.now().isoformat(),
                'sample_metrics': {
                    'total_samples': len(predictions),
                    'unique_accounts': len(predictions['account_id'].unique())
                },
                'prediction_metrics': {
                    'positive_rate': float(predictions['prediction'].mean()),
                    'prediction_counts': predictions['prediction'].value_counts().to_dict()
                },
                'probability_metrics': {
                    'mean': float(predictions['probability'].mean()),
                    'median': float(predictions['probability'].median()),
                    'std': float(predictions['probability'].std()),
                    'quantiles': {
                        '25%': float(predictions['probability'].quantile(0.25)),
                        '75%': float(predictions['probability'].quantile(0.75))
                    }
                }
            }

            self.logger.info(f"Generated health report for {len(predictions)} predictions")
            return report

        except Exception as e:
            self.logger.error(f"Error generating health report: {str(e)}")
            raise