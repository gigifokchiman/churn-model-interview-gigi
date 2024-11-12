import logging

from src.data.data_generator import give_me_some_data
from src.pipelines.config import get_config
from src.utils.logger import setup_logger
from src.utils.saving_data_minio import save_to_minio

# Configure logging
logger = setup_logger(
    name=__name__,
    level=logging.ERROR,
)

if __name__ == "__main__":
    try:
        # Load configuration from YAML file
        logger.info("Loading configuration...")
        config = get_config()
        logger.info("Configuration loaded successfully")

        # Generate and save training data
        logger.info("Generating training data...")
        random_seed = config.training_params["random_seed"]
        train_df = give_me_some_data(random_seed=random_seed)
        logger.info(f"Training data generated: {train_df.shape[0]} rows, {train_df.shape[1]} columns")

        train_path = save_to_minio(train_df, config.training_params["dataset"])
        logger.info(f"Training data saved to: {train_path}")

        # Generate and save testing data
        logger.info("Generating testing data...")
        random_seed = config.testing_params["random_seed"]
        test_df = give_me_some_data(random_seed=random_seed)
        logger.info(f"Testing data generated: {test_df.shape[0]} rows, {test_df.shape[1]} columns")

        test_path = save_to_minio(test_df, config.testing_params["dataset"])
        logger.info(f"Testing data saved to: {test_path}")

        logger.info("Data generation and saving process completed successfully")

    except Exception as e:
        logger.error(f"An error occurred during data generation and saving: {str(e)}", exc_info=True)
        raise
