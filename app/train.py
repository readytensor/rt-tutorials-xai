from typing import Tuple, Dict, Any
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
import argparse

from config import paths
from utils import (
    set_seeds,
    load_data_schema, 
    read_csv_in_directory, 
    read_json_as_dict, 
    get_validation_percentage
)
from data_management.schema_provider import BinaryClassificationSchema
from data_management.preprocess import (
    create_pipeline_and_label_encoder,
    train_pipeline_and_label_encoder,
    save_pipeline_and_label_encoder,
    transform_data,
    split_train_val, 
    handle_class_imbalance
)
from algorithm.classifier import (
    train_classifier_model,
    save_classifier,
    evaluate_classifier
)
from hyperparameter_tuning.tune import tune_hyperparameters


def load_and_split_data(val_pct: float) -> \
        Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and split the data into training and validation sets.

    Args:
        val_pct: The percentage of the data to be used for validation.

    Returns:
        A tuple containing the data schema, training split, and validation split.
    """    
    train_data = read_csv_in_directory(file_dir_path=paths.TRAIN_DIR)
    train_split, val_split = split_train_val(train_data, val_pct=val_pct)
    return train_split, val_split


def preprocess_and_balance_data(
        model_config: Dict[str, Any],
        data_schema: BinaryClassificationSchema,
        train_split: pd.DataFrame) -> Tuple[Pipeline, LabelEncoder, pd.DataFrame, pd.Series]:
    """
    Preprocess and balance the data using the provided model configuration and data schema.

    Args:
        model_config: A dictionary containing the model configuration.
        data_schema: A dictionary containing the data schema.
        train_split: A pandas DataFrame containing the train data split.

    Returns:
        A tuple containing the preprocessed pipeline, label encoder, balanced data, and balanced labels.
    """
    preprocess_pipeline, label_encoder = \
        create_pipeline_and_label_encoder(model_config, data_schema)
    transformed_data, transformed_labels = train_pipeline_and_label_encoder(
        preprocess_pipeline, label_encoder, train_split, data_schema)
    balanced_data, balanced_labels = \
        handle_class_imbalance(transformed_data, transformed_labels, random_state=0)
    return preprocess_pipeline, label_encoder, balanced_data, balanced_labels


def run_training(run_tuning: bool) -> None:
    """
    Run the training process for the binary classification model.
    """
    set_seeds(seed_value=0)

    data_schema = load_data_schema(paths.SCHEMA_DIR)
    
    model_config = read_json_as_dict(paths.MODEL_CONFIG_FILE_PATH)
    val_pct = get_validation_percentage(model_config)

    train_split, val_split = load_and_split_data(val_pct)

    preprocess_pipeline, label_encoder, balanced_train_data, balanced_train_labels = \
        preprocess_and_balance_data(model_config, data_schema, train_split)
    
    transformed_val_data, transformed_val_labels, _ = \
        transform_data(preprocess_pipeline, label_encoder, val_split, data_schema)
    
    save_pipeline_and_label_encoder(preprocess_pipeline, label_encoder,
           paths.PIPELINE_FILE_PATH, paths.LABEL_ENCODER_FILE_PATH)

    default_hyperparameters = read_json_as_dict(paths.DEFAULT_HYPERPARAMETERS_FILE_PATH)

    if run_tuning:
        hpt_specs = read_json_as_dict(paths.HPT_CONFIG_FILE_PATH)
        hyperparameters = tune_hyperparameters(
            train_X=balanced_train_data,
            train_y=balanced_train_labels,
            valid_X=transformed_val_data,
            valid_y=transformed_val_labels,
            default_hps=default_hyperparameters,
            hpt_specs=hpt_specs,
            hpt_results_dir_path=paths.HPT_OUTPUTS_DIR,
            best_hp_file_path=paths.BEST_HYPERPARAMETERS_FILE_PATH,
            is_minimize=False
        )
    else:
        hyperparameters = default_hyperparameters

    
    classifier = train_classifier_model(balanced_train_data, balanced_train_labels, hyperparameters)
    save_classifier(classifier, paths.CLASSIFIER_FILE_PATH)
    
    val_accuracy = evaluate_classifier(classifier, transformed_val_data, transformed_val_labels)
    print(f"Validation data accuracy: {round(val_accuracy, 5)}")
    print("Training completed successfully")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a binary classification model.")
    parser.add_argument(
        "-t",
        "--tune",
        action="store_true",
        help="Run hyperparameter tuning before training the model. If not set, use default hyperparameters.",
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    run_training(run_tuning=args.tune)
