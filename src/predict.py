import numpy as np
import pandas as pd
from typing import Any, List
from schema.data_schema import load_saved_schema
from utils import (
    read_json_as_dict,
    read_csv_in_directory, 
    save_dataframe_as_csv
)
from preprocessing.preprocess import (
    load_pipeline_and_target_encoder,
    transform_data
)
from prediction.predictor_model import load_predictor_model, predict_with_model
from config import paths


def create_predictions_dataframe(
        predictions_arr: np.ndarray,
        class_names: List[str],
        prediction_field_name: str,
        ids: pd.Series,
        id_field_name: str,
        return_probs: bool = False) -> pd.DataFrame:
    """
    Converts the predictions numpy array into a dataframe having the required structure.

    Performs the following transformations:
    - converts to pandas dataframe
    - adds column headers with class labels for predicted probabilities
    - inserts the id column

    Args:
        predictions_arr (np.ndarray): Predicted probabilities from predictor model.
        class_names List[str]: List of target classes (labels).
        prediction_field_name (str): Field name to use for predicted class.
        ids: ids as a numpy array for each of the samples in  predictions.
        id_field_name (str): Name to use for the id field.
        return_probs (bool, optional): If True, returns the predicted probabilities
            for each class. If False, returns the final predicted class for each
            data point. Defaults to False.

    Returns:
        Predictions as a pandas dataframe
    """
    if predictions_arr.shape[1] != len(class_names):
        raise ValueError("Length of class names does not match number of prediction columns")    
    predictions_df = pd.DataFrame(predictions_arr, columns=class_names)
    if len(predictions_arr) != len(ids):
        raise ValueError("Length of ids does not match number of predictions")   
    predictions_df.insert(0, id_field_name, ids)
    if return_probs:
        return predictions_df
    predictions_df[prediction_field_name] = \
        predictions_df[class_names].idxmax(axis=1)
    predictions_df.drop(class_names, axis=1, inplace=True)
    return predictions_df


def run_batch_predictions(
    saved_schema_path: str = paths.SAVED_SCHEMA_PATH,
    model_config_file_path: str = paths.MODEL_CONFIG_FILE_PATH,
    test_dir: str = paths.TEST_DIR,
    pipeline_file_path: str = paths.PIPELINE_FILE_PATH,
    target_encoder_file_path: str = paths.TARGET_ENCODER_FILE_PATH,
    predictor_file_path: str = paths.PREDICTOR_FILE_PATH,
    predictions_file_path: str = paths.PREDICTIONS_FILE_PATH,
) -> None:
    """
    Run batch predictions on test data, save the predicted probabilities to a CSV file.

    This function reads test data from the specified directory,
    loads the preprocessing pipeline and pre-trained predictor model,
    transforms the test data using the pipeline,
    makes predictions using the trained predictor model,
    adds ids into the predictions dataframe,
    and saves the predictions as a CSV file.

    Args:
        saved_schema_path (str): Path to the saved data schema.
        model_config_file_path (str): Path to the model configuration file.
        test_dir (str): Directory path for the test data.
        pipeline_file_path (str): Path to the saved pipeline file.
        target_encoder_file_path (str): Path to the saved target encoder file.
        predictor_file_path (str): Path to the saved predictor model file.
        predictions_file_path (str): Path where the predictions file will be saved.
    """
    data_schema = load_saved_schema(saved_schema_path)
    model_config = read_json_as_dict(model_config_file_path)
    test_data = read_csv_in_directory(file_dir_path=test_dir)
    preprocessor, target_encoder = load_pipeline_and_target_encoder(
        pipeline_file_path, target_encoder_file_path
    )
    transformed_data, _ = transform_data(preprocessor, target_encoder, test_data)
    predictor_model = load_predictor_model(predictor_file_path)
    predictions_arr = predict_with_model(
        predictor_model, transformed_data, return_probs=True)
    predictions_df = create_predictions_dataframe(
        predictions_arr,
        data_schema.target_classes,
        model_config["prediction_field_name"],
        test_data[data_schema.id],
        data_schema.id,
        return_probs=True
    )
    save_dataframe_as_csv(
        dataframe=predictions_df,
        file_path=predictions_file_path,
    )
    print("Batch predictions completed successfully")


if __name__ == "__main__":
    run_batch_predictions()
