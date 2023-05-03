import pandas as pd
from typing import Tuple
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from data_management.pipeline import get_preprocess_pipeline, save_pipeline, load_pipeline
from data_management.label_encoder import CustomLabelBinarizer, get_binary_target_encoder, save_label_encoder, load_label_encoder
from data_management.schema_provider import BinaryClassificationSchema



def create_pipeline_and_label_encoder(model_config: dict,
                                      data_schema: BinaryClassificationSchema) -> Tuple[Pipeline, CustomLabelBinarizer]:
    """
    Create the preprocessing pipeline and label encoder.

    Args:
        data_schema (BinaryClassificationSchema): An instance of the BinaryClassificationSchema.

    Returns:
        Tuple[Pipeline, CustomLabelBinarizer]: A tuple containing the preprocessing pipeline and label encoder.
    """
    preprocess_pipeline = get_preprocess_pipeline(config=model_config, data_schema=data_schema)
    label_encoder = get_binary_target_encoder(
        target_field=data_schema.target_field,
        allowed_values=data_schema.allowed_target_values,
        positive_class=data_schema.positive_class)
    return preprocess_pipeline, label_encoder


def train_pipeline_and_label_encoder(preprocess_pipeline, label_encoder,
                train_data: pd.DataFrame, data_schema: BinaryClassificationSchema) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Train the preprocessing pipeline and label encoder.

    Args:
        preprocess_pipeline: The preprocessing pipeline.
        label_encoder: The label encoder.
        train_data (pd.DataFrame): The training data as a DataFrame.
        data_schema (BinaryClassificationSchema): An instance of the BinaryClassificationSchema.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the transformed data and transformed labels.
    """
    transformed_data = preprocess_pipeline.fit_transform(train_data)
    feature_cols = [c for c in transformed_data.columns
                    if c not in [data_schema.id_field, data_schema.target_field]]
    transformed_data = transformed_data[feature_cols]
    transformed_labels = label_encoder.fit_transform(train_data[[data_schema.target_field]])    
    return transformed_data, transformed_labels


def save_pipeline_and_label_encoder(preprocess_pipeline, label_encoder,
                                    pipeline_fpath, label_encoder_fpath)-> None:
    """
    Save the preprocessing pipeline and label encoder to files.

    Args:
        preprocess_pipeline: The preprocessing pipeline.
        label_encoder: The label encoder.

    Returns:
        None
    """
    save_pipeline(pipeline=preprocess_pipeline, file_path_and_name=pipeline_fpath)
    save_label_encoder(label_encoder=label_encoder, file_path_and_name=label_encoder_fpath)


def load_pipeline_and_label_encoder(pipeline_fpath, label_encoder_fpath)-> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the preprocessing pipeline and label encoder

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the preprocessing pipeline and label encoder.
    """
    preprocess_pipeline = load_pipeline(file_path_and_name=pipeline_fpath)
    label_encoder = load_label_encoder(file_path_and_name=label_encoder_fpath)
    return preprocess_pipeline, label_encoder


def transform_data(preprocess_pipeline, label_encoder, input_data: pd.DataFrame,
                   data_schema: BinaryClassificationSchema) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Transform the input data using the preprocessing pipeline and label encoder.

    Args:
        preprocess_pipeline: The preprocessing pipeline.
        label_encoder: The label encoder.
        input_data (pd.DataFrame): The input data as a DataFrame.
        data_schema (BinaryClassificationSchema): An instance of the BinaryClassificationSchema.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: A tuple containing the transformed data and transformed labels (if available).
    """
    transformed_data = preprocess_pipeline.transform(input_data)
    feature_cols = [c for c in transformed_data.columns
                    if c not in [data_schema.id_field, data_schema.target_field]]
    ids = input_data[[data_schema.id_field]]
    transformed_data = transformed_data[feature_cols]
    if data_schema.target_field in input_data.columns:
        transformed_labels = label_encoder.transform(input_data[[data_schema.target_field]])
    else:
        transformed_labels = None
    return transformed_data, transformed_labels, ids


def handle_class_imbalance(transformed_data: pd.DataFrame, transformed_labels: pd.Series,
                           random_state: int = 0) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Handle class imbalance using SMOTE.

    Args:
        transformed_data (pd.DataFrame): The transformed data.
        transformed_labels (pd.Series): The transformed labels.
        random_state (int): The random state seed for reproducibility. Defaults to 0.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: A tuple containing the balanced data and balanced labels.
    """
    smote = SMOTE(random_state=random_state)
    balanced_data, balanced_labels = smote.fit_resample(transformed_data, transformed_labels)
    return balanced_data, balanced_labels


  
def split_train_val(data: pd.DataFrame, val_pct: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits the input data into training and validation sets based on the given percentage.

    Args:
        data (pd.DataFrame): The input data as a DataFrame.
        val_pct (float): The percentage of data to be used for the validation set.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the training and validation sets as DataFrames.
    """
    train_data, val_data = train_test_split(data, test_size=val_pct, random_state=42)
    return train_data, val_data