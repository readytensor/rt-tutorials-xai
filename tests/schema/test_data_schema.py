"""
This module contains unit tests for the BinaryClassificationSchema class of the data_schema.py module.

BinaryClassificationSchema is a class that provides an interface to interact with the schema of a binary
classification problem. It takes a schema dictionary as input and provides methods to access various properties
of the schema, such as id, target, predictors, and their properties like allowed values, example values, and
descriptions.

Each function in this module tests a specific functionality of the BinaryClassificationSchema class.
"""
import pytest
from src.schema.data_schema import BinaryClassificationSchema

def test_init():
    """
    Test the initialization of BinaryClassificationSchema class with a valid schema dictionary.
    Asserts that the properties of the schema object match the input schema dictionary.
    """
    # Given
    schema_dict = {
        "problemCategory": "binary_classification",
        "title": "Test Title",
        "description": "Test Description",
        "version": 1.0,
        "inputDataFormat": "CSV",
        "id": {"name": "Test ID"},
        "target": {"name": "Test Target", "allowedValues": ["0", "1"]},
        "predictors": [{"name": "Test Predictor", "dataType": "NUMERIC"}]
    }

    # When
    schema = BinaryClassificationSchema(schema_dict)

    # Then
    assert schema.problem_category == "binary_classification"
    assert schema.title == "Test Title"
    assert schema.description == "Test Description"
    assert schema.version == 1.0
    assert schema.input_data_format == "CSV"
    assert schema.id == "Test ID"
    assert schema.target == "Test Target"
    assert schema.allowed_target_values == ["0", "1"]
    assert schema.numeric_features == ["Test Predictor"]
    assert schema.categorical_features == []
    assert schema.features == ["Test Predictor"]
    assert schema.all_fields == ["Test ID", "Test Target", "Test Predictor"]


def test_get_allowed_values_for_categorical_feature():
    """
    Test the method to get allowed values for a categorical feature.
    Asserts that the allowed values match the input schema dictionary.
    Also tests for a ValueError when trying to get allowed values for a non-existent feature.
    """
    # Given
    schema_dict = {
        "problemCategory": "binary_classification",
        "title": "Test Title",
        "description": "Test Description",
        "version": 1.0,
        "inputDataFormat": "CSV",
        "id": {"name": "Test ID"},
        "target": {"name": "Test Target", "allowedValues": ["0", "1"]},
        "predictors": [
            {"name": "Test Predictor 1", "dataType": "NUMERIC"},
            {"name": "Test Predictor 2", "dataType": "CATEGORICAL", "allowedValues": ["A", "B"]}
        ]
    }
    schema = BinaryClassificationSchema(schema_dict)

    # When
    allowed_values = schema.get_allowed_values_for_categorical_feature("Test Predictor 2")

    # Then
    assert allowed_values == ["A", "B"]

    # When
    with pytest.raises(ValueError):
        schema.get_allowed_values_for_categorical_feature("Invalid Predictor")

    # Then: Exception is raised


def test_get_example_value_for_numeric_feature():
    """
    Test the method to get an example value for a numeric feature.
    Asserts that the example value matches the input schema dictionary.
    Also tests for a ValueError when trying to get an example value for a non-existent feature.
    """
    # Given
    schema_dict = {
        "problemCategory": "binary_classification",
        "title": "Test Title",
        "description": "Test Description",
        "version": 1.0,
        "inputDataFormat": "CSV",
        "id": {"name": "Test ID"},
        "target": {"name": "Test Target", "allowedValues": ["0", "1"]},
        "predictors": [
            {"name": "Test Predictor 1", "dataType": "NUMERIC", "example": 123.45},
            {"name": "Test Predictor 2", "dataType": "CATEGORICAL", "allowedValues": ["A", "B"]}
        ]
    }
    schema = BinaryClassificationSchema(schema_dict)

    # When
    example_value = schema.get_example_value_for_feature("Test Predictor 1")

    # Then
    assert example_value == 123.45

    # When
    with pytest.raises(ValueError):
        schema.get_example_value_for_feature("Invalid Predictor")

    # Then: Exception is raised


def test_get_description_for_id_target_and_predictors():
    """
    Test the methods to get descriptions for the id, target, and predictors.
    Asserts that the descriptions match the input schema dictionary.
    Also tests for a ValueError when trying to get a description for a non-existent feature.
    """
    # Given
    schema_dict = {
        "problemCategory": "binary_classification",
        "title": "Test Title",
        "description": "Test Description",
        "version": 1.0,
        "inputDataFormat": "CSV",
        "id": {"name": "Test ID", "description": "ID field"},
        "target": {"name": "Test Target", "description": "Target field", "allowedValues": ["0", "1"]},
        "predictors": [
            {"name": "Test Predictor 1", "dataType": "NUMERIC", "description": "Numeric predictor"},
            {"name": "Test Predictor 2", "dataType": "CATEGORICAL", "description": "Categorical predictor", "allowedValues": ["A", "B"]}
        ]
    }
    schema = BinaryClassificationSchema(schema_dict)

    # When
    id_description = schema.id_description
    target_description = schema.target_description
    predictor_1_description = schema.get_description_for_feature("Test Predictor 1")
    predictor_2_description = schema.get_description_for_feature("Test Predictor 2")

    # Then
    assert id_description == "ID field"
    assert target_description == "Target field"
    assert predictor_1_description == "Numeric predictor"
    assert predictor_2_description == "Categorical predictor"

    # When
    with pytest.raises(ValueError):
        schema.get_description_for_feature("Invalid Feature")

    # Then: Exception is raised
