import numpy as np, pandas as pd

from data_management.preprocess import (
    load_pipeline_and_label_encoder,
    transform_data
)
from data_management.label_encoder import get_class_names
from algorithm.classifier import load_classifier, predict_with_classifier
from data_management.schema_provider import BinaryClassificationSchema


class ModelServer:
    """
    Class for making batch or online predictions using a trained classifier.
    """

    def __init__(
            self,
            pipeline_path: str,
            label_encoder_path: str,
            model_artifacts_path: str,
            data_schema: BinaryClassificationSchema
    ) -> None:
        """
        Initializes a new instance of the `ModelServer` class.

        Args:
            pipeline_path: The path to the preprocessor pipeline.
            label_encoder_path: The path to the label encoder.
            model_artifacts_path: The path to the directory containing the trained model artifacts.
            data_schema: An instance of the BinaryClassificationSchema class that defines the dataset schema.
        """
        self.preprocessor, self.label_encoder = load_pipeline_and_label_encoder(
            pipeline_path, label_encoder_path)
        self.model = load_classifier(model_artifacts_path)
        self.data_schema = data_schema

    def _get_predictions(self, data: pd.DataFrame) -> np.ndarray:
        """
        Internal function to make batch predictions on the input data.

        Args:
            data (pandas.DataFrame): The input data to make predictions on.
        Returns:
            preds (numpy.ndarray): The predicted class probabilities.
        """
        # transform data - returns data, labels (if any), and ids
        transformed_data, _, ids = transform_data(self.preprocessor, self.label_encoder, data, self.data_schema)
       
        # make predictions
        predictions = predict_with_classifier(self.model, transformed_data, return_probs=True)
        return predictions, ids


    def predict_proba(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Function to make batch predictions on the input data and return predicted class probabilities.

        Args:
            data (pandas.DataFrame): The input data to make predictions on.
        Returns:
            preds_df (pandas.DataFrame): A pandas DataFrame with the input data ids and the predicted class probabilities.
        """
        preds, ids = self._get_predictions(data)
        preds_df = pd.DataFrame(ids, columns=[self.data_schema.id_field])
        class_names = get_class_names(self.label_encoder)
        preds_df[class_names] = np.round(preds, 5)
        return preds_df


    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Function to make batch predictions on the input data and return predicted classes.

        Args:
            data (pandas.DataFrame): The input data to make predictions on.
        Returns:
            preds_df (pandas.DataFrame): A pandas DataFrame with the input data ids and the predicted classes.
        """
        predicted_proba = self.predict_proba(data)
        preds_df = predicted_proba[[self.data_schema.id_field]].copy()
        class_names = get_class_names(self.label_encoder)
        preds_df["prediction"] = predicted_proba[class_names].idxmax(axis=1)
        return preds_df