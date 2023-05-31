import pytest
import pandas as pd
import numpy as np
from tempfile import TemporaryDirectory

from schema.data_schema import BinaryClassificationSchema
from preprocessing.preprocess import (
    train_pipeline_and_target_encoder,
    transform_data,
    save_pipeline_and_target_encoder,
    load_pipeline_and_target_encoder,
    handle_class_imbalance
)


# Fixture to create a sample train dataFrame for testing
@pytest.fixture
def train_split_provider():    
    """Provide a valid train_split DataFrame for testing."""
    data = pd.DataFrame(
        {
            "id": range(1, 6),
            "numeric_feature_1": [10, 20, 30, 40, 50],
            "numeric_feature_2": [1.0, -2., 3, -4, 5],
            "categorical_feature_1": ["A", "B", "C", "A", "B"],
            "categorical_feature_2": ["P", "Q", "R", "S", "T"],
            "target_field": ["A", "B", "A", "B", "A"]
        }
    )
    return data

# Fixture to create a sample validation dataFrame for testing
@pytest.fixture
def val_split_provider():
    """Provide a valid val_split DataFrame for testing."""
    data = pd.DataFrame(
        {
            "id": range(6, 11),
            "numeric_feature_1": [60, 70, 80, 90, 100],
            "numeric_feature_2": [-1.0, 2., -3, 4, -5],
            "categorical_feature_1": ["A", "B", "C", "A", "B"],
            "categorical_feature_2": ["P", "Q", "R", "S", "T"],
            "target_field": ["A", "B", "A", "B", "A"]
        }
    )
    return data


def test_train_pipeline_and_target_encoder(
        schema_provider, train_split_provider, pipeline_config_file_path):
    """Test the training of the pipeline and target encoder."""
    pipeline, target_encoder = train_pipeline_and_target_encoder(
        schema_provider, train_split_provider, pipeline_config_file_path)
    assert pipeline is not None
    assert target_encoder is not None


def test_transform_data_with_train_split(
        schema_provider, train_split_provider, pipeline_config_file_path):
    """Test if train data is properly transformed using the preprocessing pipeline and target encoder."""
    preprocess_pipeline, target_encoder = train_pipeline_and_target_encoder(
        schema_provider, train_split_provider, pipeline_config_file_path)
    transformed_inputs, transformed_targets = transform_data(
        preprocess_pipeline, target_encoder, train_split_provider)
    assert transformed_inputs is not None
    assert transformed_targets is not None
    assert len(transformed_inputs) == len(train_split_provider)
    assert len(transformed_targets) == len(train_split_provider)


def test_transform_data_with_valid_split(
        schema_provider, train_split_provider, 
        pipeline_config_file_path, val_split_provider):
    """Test if validation data is properly transformed using the preprocessing pipeline and target encoder."""
    preprocess_pipeline, target_encoder = train_pipeline_and_target_encoder(
        schema_provider, train_split_provider, pipeline_config_file_path)
    transformed_inputs, transformed_targets = transform_data(
        preprocess_pipeline, target_encoder, val_split_provider)
    assert transformed_inputs is not None
    assert transformed_targets is not None
    assert len(transformed_inputs) == len(val_split_provider)
    assert len(transformed_targets) == len(val_split_provider)


def test_save_and_load_pipeline_and_target_encoder(
        schema_provider, train_split_provider, pipeline_config_file_path):
    """
    Test that the trained pipeline and target encoder can be saved and loaded correctly, 
    and that the transformation results before and after saving/loading are the same.
    """
    preprocess_pipeline, target_encoder = train_pipeline_and_target_encoder(
        schema_provider, train_split_provider, pipeline_config_file_path)
    transformed_inputs, transformed_targets = transform_data(
        preprocess_pipeline, target_encoder, train_split_provider)
    with TemporaryDirectory() as tempdir:
        pipeline_fpath = tempdir + '/pipeline.pkl'
        target_encoder_fpath = tempdir + '/target_encoder.pkl'
        save_pipeline_and_target_encoder(
            preprocess_pipeline, target_encoder, pipeline_fpath, target_encoder_fpath)
        loaded_preprocess_pipeline, loaded_target_encoder = \
            load_pipeline_and_target_encoder(pipeline_fpath, target_encoder_fpath)
        assert loaded_preprocess_pipeline is not None
        assert loaded_target_encoder is not None
    transformed_inputs_2, transformed_targets_2 = transform_data(
        loaded_preprocess_pipeline, loaded_target_encoder, train_split_provider)
    assert transformed_inputs.equals(transformed_inputs_2)
    assert transformed_targets.equals(transformed_targets_2)
    


def test_handle_class_imbalance(
        schema_provider, train_split_provider, pipeline_config_file_path):
    """
    Test if class imbalance is properly handled using SMOTE.
    Also ensures that there is more than one class after balancing,
    and that the counts of each class are approximately equal.
    """
    preprocess_pipeline, target_encoder = train_pipeline_and_target_encoder(
        schema_provider, train_split_provider, pipeline_config_file_path)
    transformed_inputs, transformed_targets = transform_data(
        preprocess_pipeline, target_encoder, train_split_provider)
    
    balanced_data, balanced_labels = \
        handle_class_imbalance(transformed_inputs, transformed_targets)
    assert balanced_data is not None
    assert balanced_labels is not None
    assert len(np.unique(balanced_labels)) > 1  # Ensure there is more than one class after balancing

    # Ensure that the counts of each class are approximately equal
    class_counts = pd.Series(balanced_labels).value_counts()
    assert max(class_counts) - min(class_counts) <= 1