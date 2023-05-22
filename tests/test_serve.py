from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import pandas as pd
import json
import pytest
import os 

from serve import (
    app, 
    ModelResources,
    generate_unique_request_id,
    create_sample_prediction,
    create_predictions_response,
    get_model_resources
)

client = TestClient(app)


# Define a fixture for test data
@pytest.fixture
def test_request_data():
    return {
        "instances": [
            {
                "PassengerId": "879",
                "Pclass": 3,
                "Name": "Laleff, Mr. Kristo",
                "Sex": "male",
                "Age": None,
                "SibSp": 0,
                "Parch": 0,
                "Ticket": "349217",
                "Fare": 7.8958,
                "Cabin": None,
                "Embarked": "S"
            }
        ]
    }

# Define a fixture for expected response
@pytest.fixture
def test_infer_response_data():
    return {
        "status": "success",
        "message": "",
        "timestamp": "...varies...",
        "requestId": "...varies...",
        "targetClasses": [
            "0",
            "1"
        ],
        "targetDescription": "A binary variable indicating whether or not the passenger survived (0 = No, 1 = Yes).",
        "predictions": [
            {
                "sampleId": "879",
                "predictedClass": "0",
                "predictedProbabilities": [
                    0.97548,
                    0.02452
                ]
            }
        ]
    }


# Define a fixture for expected response
@pytest.fixture
def test_explanation_response_data():
    return {
        "status": "success",
        "message": "",
        "timestamp": "2023-05-22T10:51:45.860800",
        "requestId": "0ed3d0b76d",
        "targetClasses": [
            "0",
            "1"
        ],
        "targetDescription": "A binary variable indicating whether or not the passenger survived (0 = No, 1 = Yes).",
        "predictions": [
            {
                "sampleId": "879",
                "predictedClass": "0",
                "predictedProbabilities": [
                    0.92107,
                    0.07893
                ],
                "explanation": {
                    "baseline": [
                        0.57775,
                        0.42225
                    ],
                    "featureScores": {
                        "Age_na": [
                            0.05389,
                            -0.05389
                        ],
                        "Age": [
                            0.02582,
                            -0.02582
                        ],
                        "SibSp": [
                            -0.00469,
                            0.00469
                        ],
                        "Parch": [
                            0.00706,
                            -0.00706
                        ],
                        "Fare": [
                            0.05561,
                            -0.05561
                        ],
                        "Embarked_S": [
                            0.01582,
                            -0.01582
                        ],
                        "Embarked_C": [
                            0.00393,
                            -0.00393
                        ],
                        "Embarked_Q": [
                            0.00657,
                            -0.00657
                        ],
                        "Pclass_3": [
                            0.0179,
                            -0.0179
                        ],
                        "Pclass_1": [
                            0.02394,
                            -0.02394
                        ],
                        "Sex_male": [
                            0.13747,
                            -0.13747
                        ]
                    }
                }
            }
        ],
        "explanationMethod": "Shap"
    }


@pytest.fixture
def test_resources_paths():
    """Define a fixture for the paths to the test model resources."""
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    test_resources_path = os.path.join(cur_dir, "test_resources")
    # base_path = 'tests/test_resources'  # Replace this with the actual path
    return {
        'saved_schema_path': os.path.join(test_resources_path, 'schema.joblib'),
        'predictor_file_path': os.path.join(test_resources_path, 'predictor.joblib'),
        'pipeline_file_path': os.path.join(test_resources_path, 'pipeline.joblib'),
        'target_encoder_file_path': os.path.join(test_resources_path, 'target_encoder.joblib'),
        'model_config_file_path': os.path.join(test_resources_path, 'model_config.json'),
        'explainer_file_path': os.path.join(test_resources_path, 'explainer.joblib')
    }


@pytest.fixture
def test_model_resources(test_resources_paths):
    """Define a fixture for the test ModelResources object."""
    return ModelResources(**test_resources_paths)


def test_ping():
    """Test the /ping endpoint."""
    response = client.get("/ping")
    assert response.status_code == 200
    assert response.json() == {"message": "Pong!"}

@patch('serve.uuid.uuid4')
def test_generate_unique_request_id(mock_uuid):
    """Test the generate_unique_request_id function."""
    mock_uuid.return_value = MagicMock(hex='1234567890abcdef1234567890abcdef')
    assert generate_unique_request_id() == '1234567890'


@pytest.mark.parametrize(
    "sample, id_field, class_names, expected",
    [
        (
            {"id": 1, "__predicted_class": "class1", "class1": 0.7, "class2": 0.3},
            "id",
            ["class1", "class2"],
            {
                "sampleId": 1,
                "predictedClass": "class1",
                "predictedProbabilities": [0.7, 0.3]
            }
        ),
        (
            {"id": 2, "__predicted_class": "class2", "class1": 0.1, "class2": 0.9},
            "id",
            ["class1", "class2"],
            {
                "sampleId": 2,
                "predictedClass": "class2",
                "predictedProbabilities": [0.1, 0.9]
            }
        ),
        (
            {"id": 2, "__predicted_class": "0", "1": 0.01, "0": 0.99},
            "id",
            ["1", "0"],
            {
                "sampleId": 2,
                "predictedClass": "0",
                "predictedProbabilities": [0.01, 0.99]
            }
        )
    ]
)
def test_create_sample_prediction(sample, id_field, class_names, expected):
    """
    Test the create_sample_prediction function.

    Args:
        sample (dict): The sample dictionary containing the input data.
        id_field (str): The name of the field in the sample dictionary that represents the sample ID.
        class_names (list): A list of class names.
        expected (dict): The expected output dictionary.

    The function should transform the sample dictionary into the expected output dictionary structure,
    using the provided id_field, class_names, and "__predicted_class" field from the sample dictionary.
    It should return the transformed dictionary structure.
    """
    assert create_sample_prediction(sample, id_field, class_names) == expected


@patch('serve.transform_req_data_and_make_predictions')
def test_infer(mock_transform_and_predict):
    """
    Test the infer endpoint.

    Args:
       mock_transform_and_predict (MagicMock): A mock of the transform_req_data_and_make_predictions function.

    The function creates a mock request and sets the expected return value of the mock_transform_and_predict function.
    It then sends a POST request to the "/infer" endpoint with the mock request data.
    The function asserts that the response status code is 200 and the JSON response matches the expected output.
    Additionally, it checks if the mock_transform_and_predict function was called with the correct arguments.
    """
    # Creating a mock request
    mock_request = {
        "instances": [
            {
                "feature1": "value1",
                "feature2": "value2"
            }
        ]
    }

    # Define what your mock should return
    mock_transform_and_predict.return_value = pd.DataFrame(), {"status": "success", "predictions": []}

    response = client.post("/infer", data=json.dumps(mock_request))

    assert response.status_code == 200
    assert response.json() == {"status": "success", "predictions": []}
    # You can add more assertions to check if the function was called with the correct arguments
    mock_transform_and_predict.assert_called()


def test_create_predictions_response(predictions_df, schema_provider):
    """
    Test the `create_predictions_response` function.

    This test checks that the function returns a correctly structured dictionary,
    including the right keys and that the 'status' field is 'success'.
    It also checks that the 'predictions' field is a list, each element of which is a 
    dictionary with the right keys.
    Additionally, it validates the 'predictedClass' is among the 'targetClasses', and 
    the sum of 'predictedProbabilities' approximates to 1, allowing for a small numerical error.

    Args:
        predictions_df (pd.DataFrame): A fixture providing a DataFrame of model predictions.
        schema_provider (BinaryClassificationSchema): A fixture providing an instance of the BinaryClassificationSchema.

    Returns:
        None
    """
    response = create_predictions_response(predictions_df, schema_provider)

    # Check that the output is a dictionary
    assert isinstance(response, dict)

    # Check that the dictionary has the correct keys
    expected_keys = {
        "status",
        "message",
        "timestamp",
        "requestId",
        "targetClasses",
        "targetDescription",
        "predictions",
    }
    assert set(response.keys()) == expected_keys

    # Check that the 'status' field is 'success'
    assert response['status'] == 'success'

    # Check that the 'predictions' field is a list
    assert isinstance(response['predictions'], list)

    # Check that each prediction has the correct keys
    prediction_keys = {
        "sampleId",
        "predictedClass",
        "predictedProbabilities",
    }
    for prediction in response['predictions']:
        assert set(prediction.keys()) == prediction_keys
        
        # Check that 'predictedClass' is one of the 'targetClasses'
        assert prediction['predictedClass'] in response['targetClasses']
        
        # Check that 'predictedProbabilities' sum to 1 (within a small tolerance)
        assert abs(sum(prediction['predictedProbabilities']) - 1.0) < 1e-5




def test_infer_endpoint(test_model_resources: ModelResources, test_request_data, test_infer_response_data):
    """
    End-to-end integration test for the /infer endpoint of the FastAPI application.

    This test uses a TestClient from FastAPI to make a POST request to the /infer endpoint,
    and verifies that the response matches expectations.

    A ModelResources instance is created with test-specific paths using the test_model_resources fixture,
    and the application's dependency on ModelResources is overridden to use this instance for the test.

    The function sends a POST request to the "/infer" endpoint with the test_request_data
    using a TestClient from FastAPI.
    It then asserts that the response keys match the expected response keys, and compares specific
    values in the response_data with the test_infer_response_data.
    Finally, it resets the dependency_overrides after the test.

    Args:
        test_model_resources (ModelResources): The test ModelResources object.
        test_request_data (dict): The fixture for test request data.
        test_infer_response_data (dict): The fixture for expected response data.
    Returns:
        None
    """
    # Override the ModelResources dependency
    app.dependency_overrides[get_model_resources] = lambda: test_model_resources

    response = client.post("/infer", json=test_request_data)
    response_data = response.json()

    # assertions
    assert set(response_data.keys()) == set(response.json().keys())
    assert response_data["predictions"][0]["sampleId"] == test_infer_response_data["predictions"][0]["sampleId"]
    assert response_data["predictions"][0]["predictedClass"] == test_infer_response_data["predictions"][0]["predictedClass"]

    # reset the dependency_overrides after the test
    app.dependency_overrides = {}


def test_explain_endpoint(test_model_resources: ModelResources, test_request_data, test_explanation_response_data):
    """
    End-to-end integration test for the /explain endpoint of the FastAPI application.

    This test uses a TestClient from FastAPI to make a POST request to the /explain endpoint,
    and verifies that the response matches expectations.

    A ModelResources instance is created with test-specific paths using the test_model_resources fixture,
    and the application's dependency on ModelResources is overridden to use this instance for the test.

    The function sends a POST request to the "/explain" endpoint with the test_request_data
    using a TestClient from FastAPI.
    It then asserts that the response keys match the expected response keys, and compares specific
    values in the explanation_response_data with the test_explanation_response_data.
    Finally, it resets the dependency_overrides after the test.

    Args:
        test_model_resources (ModelResources): The test ModelResources object.
        test_request_data (dict): The fixture for test request data.
        test_explanation_response_data (dict): The fixture for expected explanation response data.
    Returns:
        None
    """
    # Override the ModelResources dependency
    app.dependency_overrides[get_model_resources] = lambda: test_model_resources

    response = client.post("/explain", json=test_request_data)
    explanation_response_data = response.json()

    # assertions
    assert set(explanation_response_data.keys()) == set(test_explanation_response_data.keys())
    assert explanation_response_data["predictions"][0]["sampleId"] == test_explanation_response_data["predictions"][0]["sampleId"]
    assert explanation_response_data["predictions"][0]["predictedClass"] == test_explanation_response_data["predictions"][0]["predictedClass"]
    
    # baseline assertions
    assert explanation_response_data["predictions"][0]["explanation"].get("baseline") is not None
    baseline = explanation_response_data["predictions"][0]["explanation"]["baseline"]
    assert len(baseline) == 2
    assert round(sum(baseline), 4) == 1.0000

    # explanation assertions
    assert explanation_response_data["predictions"][0]["explanation"].get("featureScores") is not None
    feature_scores = explanation_response_data["predictions"][0]["explanation"]["featureScores"]
    assert len(feature_scores) == 11

    # reset the dependency_overrides after the test
    app.dependency_overrides = {}
