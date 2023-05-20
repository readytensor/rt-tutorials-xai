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


def get_model_predictions(
        transformed_data: pd.DataFrame,
        predictor_model: Any,
        class_names: List[str],
        prediction_field_name: str,
        return_probs: bool = False) -> pd.DataFrame:
    """
    Make predictions on transformed input data using the predictor_model.

    Args:
        transformed_data (np.ndarray): Transformed data to be predicted.
        predictor_model (Any): A trained predictor model.
        class_names List[str]: List of target classes (labels)
        prediction_field_name (str): Field name to use for predicted class
        return_probs (bool, optional): If True, returns the predicted probabilities for each class.
                                       If False, returns the final predicted class for each data point.
                                       Defaults to False.

    Returns:
        pd.DataFrame: A DataFrame with the same length as the input data. Contains either the
                      predicted probabilities or the final prediction for each data point.
    """
    predictions_arr = predict_with_model(
        predictor_model, transformed_data, return_probs=True)
    predictions_df = pd.DataFrame(predictions_arr, columns=class_names)
    if return_probs:
        return predictions_df
    predictions_df[prediction_field_name] = \
        predictions_df[class_names].idxmax(axis=1)
    predictions_df.drop(class_names, axis=1, inplace=True)
    return predictions_df


def add_ids_to_predictions(
        input_data: pd.DataFrame,
        predictions: pd.DataFrame,
        id_field_name: str):
    """
    Insert the id column in the predictions dataframe.

    Takes the id column from given input data and inserts it into the predictions dataframe.
    Assumes the order was mained in the predictions.

    Args:
         input_data (pd.Dataframe): Input data for predictions.
         predictions (pd.Dataframe): Predictions dataframe.
         id_field_name (str): Name to use for the id field.

    Returns:
        pd.DataFrame: The predictions dataframe with ids
    """
    predictions.insert(0, id_field_name, input_data[id_field_name].values)
    return predictions


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
    predictions_df = get_model_predictions(
        transformed_data,
        predictor_model,
        data_schema.allowed_target_values,
        model_config["prediction_field_name"],
        return_probs=True
    )
    predictions_df_with_ids = add_ids_to_predictions(
        test_data, predictions_df, data_schema.id)

    save_dataframe_as_csv(
        dataframe=predictions_df_with_ids,
        file_path=predictions_file_path,
    )
    print("Batch predictions completed successfully")



if __name__ == "__main__":
    run_batch_predictions()
