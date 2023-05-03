from typing import List
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin


class CustomLabelBinarizer(BaseEstimator, TransformerMixin):
    """ Binarizes the target variable to 0/1 values. """
    def __init__(self, target_field:str, allowed_values: List[str], positive_class: str) -> None:
        """
        Initializes a new instance of the `CustomLabelBinarizer` class.

        Args:
            target_field: str
                Name of the target field.
            allowed_values: List[str]
                Class labels in a list.
            positive_class: str
                Name of the target class.
        """

        if positive_class not in allowed_values:
            raise ValueError(f"Positive class {positive_class} not in allowed values {allowed_values}")

        self.target_field = target_field
        self.positive_class = str(positive_class)
        self.negative_class = str([value for value in allowed_values if value != positive_class][0])
        self.classes = [self.negative_class, self.positive_class]
        self.class_encoding = {self.negative_class:0, self.positive_class:1}

    def fit(self, data):
        """
        No-op.

        Returns:
            self
        """   
        return self

    def transform(self, data):
        """
        Transform the data.

        Args:
            data: pandas DataFrame - data to transform
        Returns:
            pandas DataFrame - transformed data
        """
        data = data.copy()
        if self.target_field in data.columns:
            observed_classes = set(data[self.target_field].astype(str))
            if len(observed_classes.intersection(self.classes)) != 2:
                raise ValueError(f"Observed classes in target {list(observed_classes)} do not match given allowed values for target: {self.allowed_values}")
            data[self.target_field] = data[self.target_field].apply(str).map(self.class_encoding)
        return data


def get_binary_target_encoder(target_field:str, allowed_values: List[str], positive_class: str) -> LabelEncoder:
    """Create a LabelEncoder based on the data_schema.

    The positive class will be encoded as 1, and the negative class will be encoded as 0.

    Args:
        target_field: Name of the target field.
        allowed_values: A list of allowed target variable values.
        positive_class: The target value representing the positive class.

    Returns:
        A SciKit-Learn LabelEncoder instance.
    """
    # Create a LabelEncoder instance and fit it with the desired class order
    encoder = CustomLabelBinarizer(
        target_field=target_field,
        allowed_values=allowed_values,
        positive_class=positive_class
    )
    return encoder


def get_class_names(label_encoder: LabelEncoder) -> List[str]:
    """Get the names of the classes for the target variable.
    We get this from the label encoder because the label encoder sorts
    the classes with the positive class second in the list.

    Args:
        label_encoder: A CustomLabelBinarizer instance.

    Returns:
        A list of class names for the target variable.
    """
    class_names = label_encoder.classes
    return class_names


def save_label_encoder(label_encoder: LabelEncoder, file_path_and_name: str) -> None:
    """Save a fitted label encoder to a file using joblib.

    Args:
        label_encoder: A fitted LabelEncoder instance.
        file_path_and_name (str): The filepath to save the LabelEncoder to.
    """
    joblib.dump(label_encoder, file_path_and_name)


def load_label_encoder(file_path_and_name: str) -> LabelEncoder:
    """Load the fitted label encoder from the given path.

    Args:
        file_path_and_name: Path to the saved label encoder.

    Returns:
        Fitted label encoder.
    """
    return joblib.load(file_path_and_name)
