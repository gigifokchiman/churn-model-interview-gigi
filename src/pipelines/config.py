from dataclasses import dataclass
from typing import Dict, Any, Optional
import os
import yaml
from pathlib import Path


@dataclass
class ModelConfig:
    model_params: Dict[str, Any]
    training_params: Dict[str, Any]
    testing_params: Dict[str, Any]
    inference_params: Dict[str, Any]

    @classmethod
    def from_yaml(cls, config_path: str) -> 'ModelConfig':
        """
        Create a ModelConfig instance from a YAML file.

        Args:
            config_path (str): Path to the YAML configuration file

        Returns:
            ModelConfig: Configuration object

        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If YAML parsing fails
            ValueError: If required configuration sections are missing
        """
        config_file = Path(config_path)

        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)

            if not isinstance(config, dict):
                raise ValueError("Configuration file must contain a YAML dictionary")

            # Validate required sections
            required_sections = ['model_params', 'training_params',
                                 'testing_params', 'inference_params']
            missing_sections = [section for section in required_sections
                                if section not in config]

            if missing_sections:
                raise ValueError(f"Missing required configuration sections: {missing_sections}")

            return cls(
                model_params=config['model_params'],
                training_params=config['training_params'],
                testing_params=config['testing_params'],
                inference_params=config['inference_params']
            )

        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML configuration: {str(e)}")


def get_config(config_path: Optional[str] = None) -> ModelConfig:
    """
    Get configuration from specified path or default location.

    Args:
        config_path (Optional[str]): Path to configuration file. 
                                   If None, uses default path.

    Returns:
        ModelConfig: Configuration object

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If configuration is invalid
    """
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), "config.yaml")

    return ModelConfig.from_yaml(config_path)
