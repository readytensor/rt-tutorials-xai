from typing import List, Dict, Tuple


class BinaryClassificationSchema:
    """
    A class for loading and providing access to a binary classification schema. This class allows
    users to work with a generic schema for binary classification problems, enabling them to
    create algorithm implementations that are not hardcoded to specific feature names. The class
    provides methods to retrieve information about the schema, such as the ID field, target field,
    allowed values for the target field, and details of the predictor fields (categorical and numeric).
    This makes it easier to preprocess and manipulate the input data according to the schema,
    regardless of the specific dataset used.
    """

    def __init__(self, schema_dict: dict) -> None:
        """
        Initializes a new instance of the `BinaryClassificationSchema` class
        and using the schema dictionary.

        Args:
            schema_dict (dict): The python dictionary of schema.
        """
        self.schema = schema_dict
        self._numeric_features, self._categorical_features = self._get_features()

    def _get_features(self) -> Tuple[List[str], List[str]]:
        """
        Returns the feature names of numeric and categorical data types.

        Returns:
            Tuple[List[str], List[str]]: The list of numeric feature names, and the list of categorical feature names.
        """
        fields = self.schema["predictorFields"]
        numeric_features = [f["name"] for f in fields if f["dataType"] == "NUMERIC"]
        categorical_features = [f["name"] for f in fields if f["dataType"] == "CATEGORICAL"]
        return numeric_features, categorical_features

    @property
    def id_field(self) -> str:
        """
        Gets the name of the ID field.

        Returns:
            str: The name of the ID field.
        """
        return self.schema["idField"]["name"]

    @property
    def target_field(self) -> str:
        """
        Gets the name of the target field.

        Returns:
            str: The name of the target field.
        """
        return self.schema["targetField"]["name"]

    @property
    def allowed_target_values(self) -> List[str]:
        """
        Gets the allowed values for the target field.

        Returns:
            List[str]: The list of allowed values for the target field.
        """
        return self.schema["targetField"]["allowedValues"]    

    @property
    def positive_class(self) -> str:
        """
        Gets the positive class for the target field.

        Returns:
            str: The positive class for the target field.
        """
        return self.schema["targetField"]["positiveClass"]

    @property
    def target_description(self) -> str:
        """
        Gets the description for the target field.

        Returns:
            str: The description for the target field.
        """
        return self.schema["targetField"].get("description", "No description for target available.")

    @property
    def numeric_features(self) -> List[str]:
        """
        Gets the names of the numeric features.

        Returns:
            List[str]: The list of numeric feature names.
        """
        return self._numeric_features

    @property
    def categorical_features(self) -> List[str]:
        """
        Gets the names of the categorical features.

        Returns:
            List[str]: The list of categorical feature names.
        """
        return self._categorical_features
    
    @property
    def allowed_categorical_values(self) -> Dict[str, List[str]]:
        """
        Gets the allowed values for the categorical features.

        Returns:
            Dict[str, List[str]]: A dictionary of categorical feature names and their corresponding allowed values.
        """
        fields = self.schema["predictorFields"]
        allowed_values = {}
        for field in fields:
            if field["dataType"] == "CATEGORICAL":
                allowed_values[field["name"]] = field["allowedValues"]
        return allowed_values

    def get_allowed_values_for_categorical_feature(self, feature_name: str) -> List[str]:
        """
        Gets the allowed values for a single categorical feature.

        Args:
            feature_name (str): The name of the categorical feature.

        Returns:
            List[str]: The list of allowed values for the specified categorical feature.
        """
        fields = self.schema["predictorFields"]
        for field in fields:
            if field["dataType"] == "CATEGORICAL" and field["name"] == feature_name:
                return field["allowedValues"]
        raise ValueError(f"Categorical feature '{feature_name}' not found in the schema.")
    
    def get_description_for_feature(self, feature_name: str) -> str:
        """
        Gets the description for a single feature.

        Args:
            feature_name (str): The name of the feature.

        Returns:
            str: The description for the specified feature.
        """
        fields = self.schema["predictorFields"]
        for field in fields:
            if field["name"] == feature_name:
                return field.get("description", "No description for feature available.")
        raise ValueError(f"Feature '{feature_name}' not found in the schema.")
    
    def get_example_value_for_feature(self, feature_name: str) -> List[str]:
        """
        Gets the example value for a single feature.

        Args:
            feature_name (str): The name of the feature.

        Returns:
            List[str]: The example values for the specified feature.
        """       

        fields = self.schema["predictorFields"]
        for field in fields:
            if field["name"] == feature_name:
                if field["dataType"] == "NUMERIC":
                    return field.get("example", 0.0)
                elif field["dataType"] == "CATEGORICAL":
                    return field["allowedValues"][0]
                else: raise ValueError(f"Invalid data type for Feature '{feature_name}' found in the schema.")
        raise ValueError(f"Feature '{feature_name}' not found in the schema.")

    @property
    def features(self) -> List[str]:
        """
        Gets the names of all the features.

        Returns:
            List[str]: The list of all feature names (numeric and categorical).
        """
        return self.numeric_features + self.categorical_features

    @property
    def all_fields(self) -> List[str]:
        """
        Gets the names of all the fields.

        Returns:
            List[str]: The list of all field names (ID field, target field, and all features).
        """
        return [self.id_field, self.target_field] + self.features


