import pytest
import os
from src.train import run_training


def test_run_training(
        tmpdir, 
        input_schema_dir, 
        model_config_file_path,
        train_dir, 
        pipeline_config_file_path,
        default_hyperparameters_file_path,
        hpt_specs_file_path,
        explainer_config_file_path
    ):
    """Test the run_training function to make sure it produces the required artifacts"""
    # Create temporary paths
    saved_schema_path = str(tmpdir.join('saved_schema.json'))
    pipeline_file_path = str(tmpdir.join('pipeline.joblib'))
    target_encoder_file_path = str(tmpdir.join('target_encoder.joblib'))
    predictor_file_path = str(tmpdir.join('predictor.joblib'))
    hpt_results_file_path = str(tmpdir.join('hpt_results.csv'))
    explainer_file_path = str(tmpdir.join('explainer.joblib'))

    # Run the training process without tuning
    run_training(
        input_schema_dir=input_schema_dir,
        saved_schema_path=saved_schema_path,
        model_config_file_path=model_config_file_path,
        train_dir=train_dir,
        pipeline_config_file_path=pipeline_config_file_path,
        pipeline_file_path=pipeline_file_path,
        target_encoder_file_path=target_encoder_file_path,
        predictor_file_path=predictor_file_path,
        default_hyperparameters_file_path=default_hyperparameters_file_path,
        run_tuning=False,
        explainer_config_file_path=explainer_config_file_path,
        explainer_file_path=explainer_file_path
    )

    # Assert that the model artifacts are saved in the correct paths
    assert os.path.isfile(saved_schema_path)
    assert os.path.isfile(pipeline_file_path)
    assert os.path.isfile(target_encoder_file_path)
    assert os.path.isfile(predictor_file_path)
    assert os.path.isfile(explainer_file_path)

    # Run the training process with tuning
    run_training(
        input_schema_dir=input_schema_dir,
        saved_schema_path=saved_schema_path,
        model_config_file_path=model_config_file_path,
        train_dir=train_dir,
        pipeline_config_file_path=pipeline_config_file_path,
        pipeline_file_path=pipeline_file_path,
        target_encoder_file_path=target_encoder_file_path,
        predictor_file_path=predictor_file_path,
        default_hyperparameters_file_path=default_hyperparameters_file_path,
        run_tuning=True,
        hpt_specs_file_path=hpt_specs_file_path,
        hpt_results_file_path=hpt_results_file_path,
        explainer_config_file_path=explainer_config_file_path,
        explainer_file_path=explainer_file_path
    )

    # Assert that the model artifacts are saved in the correct paths
    assert os.path.isfile(saved_schema_path)
    assert os.path.isfile(pipeline_file_path)
    assert os.path.isfile(target_encoder_file_path)
    assert os.path.isfile(predictor_file_path)
    assert os.path.isfile(hpt_results_file_path)
    assert os.path.isfile(explainer_file_path)
