import pytest
import pandas as pd
import numpy as np
from minio import Minio
from datetime import datetime

from src.utils.saving_data_minio import save_to_minio, read_from_minio


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame({
        'id': range(1, 4),
        'name': ['Alice', 'Bob', 'Charlie'],
        'value': [10.5, 20.0, 30.7]
    })


@pytest.fixture
def minio_test_config():
    """MinIO configuration for testing."""
    return {
        'bucket': 'data-staging',
        'minio_host': 'localhost:9000',
        'minio_access_key': 'minio',
        'minio_secret_key': 'minio123',
        'secure': False
    }


def test_save_and_read_minio(sample_dataframe, minio_test_config):
    """Test saving and reading DataFrame to/from MinIO."""
    filename = 'test_data.csv'

    # Test saving
    save_result = save_to_minio(
        df=sample_dataframe,
        filename=filename,
        **minio_test_config
    )
    assert save_result

    # Test reading
    read_df = read_from_minio(
        filename=filename,
        **minio_test_config
    )

    # Compare DataFrames
    pd.testing.assert_frame_equal(sample_dataframe, read_df)


def test_save_minio_empty_dataframe(minio_test_config):
    """Test saving an empty DataFrame."""
    # Create empty DataFrame with columns
    empty_df = pd.DataFrame(columns=['col1', 'col2'])
    filename = 'empty_test.csv'

    save_result = save_to_minio(
        df=empty_df,
        filename=filename,
        **minio_test_config
    )
    assert save_result == True

    read_df = read_from_minio(
        filename=filename,
        **minio_test_config
    )
    pd.testing.assert_frame_equal(empty_df, read_df)


def test_save_minio_invalid_credentials(sample_dataframe):
    """Test saving with invalid credentials."""
    with pytest.raises(Exception) as exc_info:
        save_to_minio(
            df=sample_dataframe,
            filename='test.csv',
            minio_access_key='invalid',
            minio_secret_key='invalid'
        )
    assert "Error saving file to MinIO" in str(exc_info.value)


def test_read_minio_nonexistent_file(minio_test_config):
    """Test reading a non-existent file."""
    with pytest.raises(Exception) as exc_info:
        read_from_minio(
            filename='nonexistent.csv',
            **minio_test_config
        )
    assert "Error reading file from MinIO" in str(exc_info.value)


@pytest.fixture(autouse=True)
def cleanup_minio(minio_test_config):
    """Cleanup test files after each test."""
    yield
    try:
        client = Minio(
            minio_test_config['minio_host'],
            access_key=minio_test_config['minio_access_key'],
            secret_key=minio_test_config['minio_secret_key'],
            secure=minio_test_config['secure']
        )

        # List and remove all objects in test bucket
        objects = client.list_objects(minio_test_config['bucket'], prefix='test')
        for obj in objects:
            client.remove_object(minio_test_config['bucket'], obj.object_name)
    except:
        pass
