import os
import pandas as pd

from predict import run_batch_predictions
from train import run_training


def test_integration_run_batch_predictions_without_hpt(
        tmpdir,
        input_schema_dir,
        model_config_file_path,
        train_dir,
        pipeline_config_file_path,
        test_dir,
        sample_test_data,
        schema_provider,
        default_hyperparameters_file_path):
    """
    Integration test for the run_batch_predictions function.

    This test simulates the full prediction pipeline, from reading the test data
    to saving the final predictions. The function is tested to ensure that it
    reads in the test data correctly, properly transforms the data, makes the
    predictions, and saves the final predictions in the correct format and location.

    The test also checks that the function handles various file and directory paths
    correctly and that it can handle variations in the input data.

    The test uses a temporary directory provided by the pytest's tmpdir fixture to
    avoid affecting the actual file system.

    Args:
        tmpdir (LocalPath): Temporary directory path provided by pytest's tmpdir fixture.
        input_schema_dir (str): Directory path to the input data schema.
        model_config_file_path (str): Path to the model configuration file.
        train_dir (str): Directory path to the training data.
        pipeline_config_file_path (str): Directory path to the pipeline config file.
        test_dir (str): Directory path to the test data.
        sample_test_data (pd.DataFrame): Sample DataFrame for testing.
        schema_provider (Any): Loaded schema provider.
        default_hyperparameters_file_path (str): Path to default hyperparameters.
        hpt_config_file_path (str): Path to HPT config file.
        explainer_config_file_path (str): Path to explainer config file.
    """
    # Create temporary paths for training
    saved_schema_path = str(tmpdir.join('saved_schema.json'))
    pipeline_file_path = str(tmpdir.join('pipeline.joblib'))
    target_encoder_file_path = str(tmpdir.join('target_encoder.joblib'))
    predictor_file_path = str(tmpdir.join('predictor.joblib'))

    # Run the training process without hyperparameter tuning
    run_training(
        input_schema_dir=input_schema_dir,
        saved_schema_path=saved_schema_path,
        model_config_file_path=model_config_file_path,
        train_dir=train_dir,
        pipeline_config_file_path=pipeline_config_file_path,
        pipeline_file_path=pipeline_file_path,
        target_encoder_file_path=target_encoder_file_path,
        predictor_file_path=predictor_file_path,
        default_hyperparameters_file_path=default_hyperparameters_file_path
    )

    # Create temporary paths for prediction
    predictions_file_path = str(tmpdir.join('predictions.csv'))

    # Run the prediction process
    run_batch_predictions(
        saved_schema_path=saved_schema_path,
        model_config_file_path=model_config_file_path,
        test_dir=test_dir,
        pipeline_file_path=pipeline_file_path,
        target_encoder_file_path=target_encoder_file_path,
        predictor_file_path=predictor_file_path,
        predictions_file_path=predictions_file_path
    )

    # Assert that the predictions file is saved in the correct path
    assert os.path.isfile(predictions_file_path)

    # Load predictions and validate the format
    predictions_df = pd.read_csv(predictions_file_path)

    # Assert that predictions dataframe has the right columns
    assert schema_provider.id in predictions_df.columns
    for class_ in schema_provider.target_classes:
        assert class_ in predictions_df.columns

    # Assert that the number of rows in the predictions matches the number of rows in the test data
    assert len(predictions_df) == len(sample_test_data)
