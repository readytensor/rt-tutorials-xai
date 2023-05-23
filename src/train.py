
import argparse
from config import paths
from schema.data_schema import load_json_data_schema, save_schema
from config import paths
from utils import (
    set_seeds,
    read_csv_in_directory,
    split_train_val,
    read_json_as_dict
)
from preprocessing.preprocess import (
    train_pipeline_and_target_encoder,
    transform_data,
    save_pipeline_and_target_encoder,
    handle_class_imbalance
)
from prediction.predictor_model import (
    train_predictor_model,
    evaluate_predictor_model,
    save_predictor_model
)
from hyperparameter_tuning.tuner import tune_hyperparameters
from xai.explainer import fit_and_save_explainer


def run_training(
        input_schema_dir: str = paths.INPUT_SCHEMA_DIR,
        saved_schema_path: str = paths.SAVED_SCHEMA_PATH,
        model_config_file_path: str = paths.MODEL_CONFIG_FILE_PATH,
        train_dir: str = paths.TRAIN_DIR,
        pipeline_config_file_path: str = paths.PREPROCESSING_CONFIG_FILE_PATH,
        pipeline_file_path: str = paths.PIPELINE_FILE_PATH,
        target_encoder_file_path: str = paths.TARGET_ENCODER_FILE_PATH,
        predictor_file_path: str = paths.PREDICTOR_FILE_PATH,
        default_hyperparameters_file_path: str = paths.DEFAULT_HYPERPARAMETERS_FILE_PATH,
        run_tuning: bool = False,
        hpt_specs_file_path: str = paths.HPT_CONFIG_FILE_PATH,
        hpt_results_file_path: str = paths.HPT_RESULTS_FILE_PATH,
        explainer_config_file_path: str = paths.EXPLAINER_CONFIG_FILE_PATH,
        explainer_file_path: str = paths.EXPLAINER_FILE_PATH) -> None:
    """
    Run the training process and saves model artifacts

    Args:
        input_schema_dir (str, optional): The directory path of the input schema.
        saved_schema_path (str, optional): The path where to save the schema.
        model_config_file_path (str, optional): The path of the model configuration file.
        train_dir (str, optional): The directory path of the train data.
        pipeline_config_file_path (str, optional): The path of the preprocessing configuration file.
        pipeline_file_path (str, optional): The path where to save the pipeline.
        target_encoder_file_path (str, optional): The path where to save the target encoder.
        predictor_file_path (str, optional): The path where to save the predictor model.
        default_hyperparameters_file_path (str, optional): The path of the default hyperparameters file.
        run_tuning (bool, optional): Whether to run hyperparameter tuning. Default is False.
        hpt_specs_file_path (str, optional): The path of the configuration file for hyperparameter tuning.
        hpt_results_file_path (str, optional): The path where to save the HPT results.
        explainer_config_file_path (str, optional): The path of the explainer configuration file.
        explainer_file_path (str, optional): The path where to save the explainer.
    Returns:
        None
    """
    # load and save schema
    data_schema = load_json_data_schema(input_schema_dir)
    save_schema(schema=data_schema, output_path=saved_schema_path)

    # load model config
    model_config = read_json_as_dict(model_config_file_path)
    set_seeds(seed_value=model_config["seed_value"])

    # load train data
    train_data = read_csv_in_directory(file_dir_path=train_dir)

    # split train data into training and validation sets
    train_split, val_split = split_train_val(
        train_data, val_pct=model_config["validation_split"])

    # fit and transform using pipeline and target encoder, then save them
    pipeline, target_encoder = train_pipeline_and_target_encoder(
        data_schema, train_split, pipeline_config_file_path)
    transformed_train_inputs, transformed_train_targets = transform_data(
        pipeline, target_encoder, train_split)
    transformed_val_inputs, transformed_val_labels = transform_data(
        pipeline, target_encoder, val_split)
    balanced_train_inputs, balanced_train_labels = \
        handle_class_imbalance(transformed_train_inputs,
                               transformed_train_targets)
    save_pipeline_and_target_encoder(
        pipeline, target_encoder,
        pipeline_file_path,
        target_encoder_file_path)
    
    # hyperparameter tuning + training the model
    if run_tuning:
        tuned_hyperparameters = tune_hyperparameters(
            train_X=balanced_train_inputs,
            train_y=balanced_train_labels,
            valid_X=transformed_val_inputs,
            valid_y=transformed_val_labels,
            hpt_results_file_path=hpt_results_file_path,
            is_minimize=False,
            default_hyperparameters_file_path = default_hyperparameters_file_path,
            hpt_specs_file_path = hpt_specs_file_path)
        predictor = train_predictor_model(
            balanced_train_inputs,
            balanced_train_labels,
            hyperparameters=tuned_hyperparameters)
    else:
        # uses default hyperparameters to train model
        default_hyperparameters = read_json_as_dict(default_hyperparameters_file_path)
        predictor = train_predictor_model(
            balanced_train_inputs, balanced_train_labels, default_hyperparameters)

    # save predictor model
    save_predictor_model(predictor, predictor_file_path)

    # calculate and print validation accuracy
    val_accuracy = evaluate_predictor_model(
        predictor, transformed_val_inputs, transformed_val_labels)
    print("Validation accuracy:", round(val_accuracy, 3))

    # fit and save explainer
    fit_and_save_explainer(
        transformed_train_inputs,
        explainer_config_file_path,
        explainer_file_path)

    print("Training completed successfully")


def parse_arguments() -> argparse.Namespace:
    """Parse the command line argument that indicates if user wants to run hyperparameter tuning."""
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
