import pandas as pd
from typing import Tuple, Any, Union
from imblearn.over_sampling import SMOTE

from preprocessing.pipeline import (
    get_preprocess_pipeline,
    train_pipeline,
    transform_inputs,
    save_pipeline,
    load_pipeline
)
from preprocessing.target_encoder import (
    get_target_encoder,
    train_target_encoder,
    transform_targets,
    save_target_encoder,
    load_target_encoder
)
from utils import read_json_as_dict

def train_pipeline_and_target_encoder(
        data_schema: Any,
        train_split: pd.DataFrame,
        pipeline_config_file_path: str
        ) -> Tuple[Any, Any]:
    """
    Train the pipeline and target encoder

    Args:
        data_schema (Any): A dictionary containing the data schema.
        train_split (pd.DataFame): A pandas DataFrame containing the train data split.

    Returns:
        A tuple containing the pipeline and target encoder.
    """
    pipeline_config = read_json_as_dict(pipeline_config_file_path)
    
    # create input trnasformation pipeline and target encoder
    preprocess_pipeline = get_preprocess_pipeline(
        data_schema=data_schema, pipeline_config=pipeline_config)
    target_encoder = get_target_encoder(data_schema=data_schema)

    # train pipeline and target encoder
    trained_pipeline = train_pipeline(preprocess_pipeline, train_split)
    trained_target_encoder = train_target_encoder(target_encoder, train_split)

    return trained_pipeline, trained_target_encoder


def transform_data(
        preprocess_pipeline: Any,
        target_encoder: Any,
        data: pd.DataFrame) \
    -> Tuple[pd.DataFrame, Union[pd.Series, None]]:
    """
    Transform the data using the preprocessing pipeline and target encoder.

    Args:
        preprocess_pipeline (Any): The preprocessing pipeline.
        target_encoder (Any): The target encoder.
        data (pd.DataFrame): The input data as a DataFrame (targets may be included).

    Returns:
        Tuple[pd.DataFrame, Union[pd.Series, None]]: A tuple containing the transformed data and transformed targets;
            transformed targets are None if the data does not contain targets.
    """    
    transformed_inputs = transform_inputs(preprocess_pipeline, data)
    transformed_targets = transform_targets(target_encoder, data)
    return transformed_inputs, transformed_targets


def save_pipeline_and_target_encoder(preprocess_pipeline: Any, target_encoder: Any,
                                    pipeline_fpath: str, target_encoder_fpath: str)-> None:
    """
    Save the preprocessing pipeline and target encoder to files.

    Args:
        preprocess_pipeline: The preprocessing pipeline.
        target_encoder: The target encoder.
        pipeline_fpath (str): full path where the pipeline is to be saved
        label_encoder_fpath (str): full path where the label encoder is to be saved

    """
    save_pipeline(pipeline=preprocess_pipeline, file_path_and_name=pipeline_fpath)
    save_target_encoder(target_encoder=target_encoder, file_path_and_name=target_encoder_fpath)


def load_pipeline_and_target_encoder(
        pipeline_fpath: str, target_encoder_fpath: str)-> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the preprocessing pipeline and target encoder

    Args:
        pipeline_fpath (str): full path where the pipeline is saved
        target_encoder_fpath (str): full path where the target encoder is saved

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the preprocessing pipeline and target encoder.
    """
    preprocess_pipeline = load_pipeline(file_path_and_name=pipeline_fpath)
    target_encoder = load_target_encoder(file_path_and_name=target_encoder_fpath)
    return preprocess_pipeline, target_encoder


def handle_class_imbalance(
        transformed_data: pd.DataFrame,
        transformed_labels: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Handle class imbalance using SMOTE.

    Args:
        transformed_data (pd.DataFrame): The transformed data.
        transformed_labels (pd.Series): The transformed labels.
        random_state (int): The random state seed for reproducibility. Defaults to 0.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: A tuple containing the balanced data and balanced labels.
    """
    # Adjust k_neighbors parameter for SMOTE
    # set k_neighbors to be the smaller of two values:
    #       1 and,
    #       the number of instances in the minority class minus one
    k_neighbors = min(1, sum(transformed_labels==min(transformed_labels.value_counts().index))-1)
    smote = SMOTE(k_neighbors=k_neighbors, random_state=0)
    balanced_data, balanced_labels = smote.fit_resample(transformed_data, transformed_labels)
    return balanced_data, balanced_labels