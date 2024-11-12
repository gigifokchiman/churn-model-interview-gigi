from minio import Minio
import pandas as pd
from io import BytesIO
from typing import Optional, Union
import logging

logger = logging.getLogger(__name__)


def save_to_minio(
    df: pd.DataFrame,
    filename: str,
    bucket: str = "data-staging",
    minio_host: str = "localhost:9000",
    minio_access_key: str = "minio",
    minio_secret_key: str = "minio123",
    secure: bool = False
) -> str:
    """
    Save a pandas DataFrame as CSV to MinIO storage.

    Args:
        df (pd.DataFrame): DataFrame to save
        filename (str): Name of the file to save (include .csv extension)
        bucket (str): MinIO bucket name
        minio_host (str): MinIO host address
        minio_access_key (str): MinIO access key
        minio_secret_key (str): MinIO secret key
        secure (bool): Whether to use HTTPS

    Returns:
        str: Path to the saved file in MinIO

    Raises:
        Exception: If there's an error saving the file
    """
    try:
        logger.info(f"Initializing MinIO client for host: {minio_host}")
        # Initialize MinIO client
        client = Minio(
            minio_host,
            access_key=minio_access_key,
            secret_key=minio_secret_key,
            secure=secure
        )

        logger.info(f"Converting DataFrame with shape {df.shape} to CSV")
        # Convert DataFrame to CSV bytes
        csv_bytes = df.to_csv(index=False).encode('utf-8')
        csv_buffer = BytesIO(csv_bytes)

        logger.info(f"Uploading file {filename} to bucket {bucket}")
        # Upload to MinIO
        client.put_object(
            bucket,
            filename,
            csv_buffer,
            length=len(csv_bytes),
            content_type='application/csv'
        )

        path = f"minio://{bucket}/{filename}"
        logger.info(f"Successfully saved file to: {path}")
        return path

    except Exception as e:
        logger.error(f"Error saving file to MinIO: {str(e)}", exc_info=True)
        raise Exception(f"Error saving file to MinIO: {str(e)}")


def read_from_minio(
    filename: str,
    bucket: str = "data-staging",
    minio_host: str = "localhost:9000",
    minio_access_key: str = "minio",
    minio_secret_key: str = "minio123",
    secure: bool = False
) -> pd.DataFrame:
    """
    Read a CSV file from MinIO storage into a pandas DataFrame.

    Args:
        filename (str): Name of the file to read (include .csv extension)
        bucket (str): MinIO bucket name
        minio_host (str): MinIO host address
        minio_access_key (str): MinIO access key
        minio_secret_key (str): MinIO secret key
        secure (bool): Whether to use HTTPS

    Returns:
        pd.DataFrame: DataFrame containing the CSV data

    Raises:
        Exception: If there's an error reading the file
    """
    try:
        logger.info(f"Initializing MinIO client for host: {minio_host}")
        # Initialize MinIO client
        client = Minio(
            minio_host,
            access_key=minio_access_key,
            secret_key=minio_secret_key,
            secure=secure
        )

        logger.info(f"Reading file {filename} from bucket {bucket}")
        # Get object data
        response = client.get_object(bucket, filename)

        logger.info("Converting data to DataFrame")
        # Read into DataFrame
        df = pd.read_csv(BytesIO(response.read()))

        logger.info(f"Successfully read DataFrame with shape: {df.shape}")
        return df

    except Exception as e:
        logger.error(f"Error reading file from MinIO: {str(e)}", exc_info=True)
        raise Exception(f"Error reading file from MinIO: {str(e)}")
