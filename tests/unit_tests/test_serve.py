from fastapi.testclient import TestClient
from unittest.mock import patch
import pandas as pd
import json
import pytest

from serve import  create_app


@pytest.fixture
def app(model_resources):
    """Define a fixture for the test app."""
    return TestClient(create_app(model_resources))


@pytest.fixture
def sample_request_data():
    # Define a fixture for test data
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

@pytest.fixture
def sample_response_data():
    # Define a fixture for expected response
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


def test_ping(app):
    """Test the /ping endpoint."""
    response = app.get("/ping")
    assert response.status_code == 200
    assert response.json() == {"message": "Pong!"}


@patch('serve.transform_req_data_and_make_predictions')
def test_infer_endpoint(mock_transform_and_predict, app, sample_request_data):
    """
    Test the infer endpoint.

    Args:
        mock_transform_and_predict (MagicMock): A mock of the transform_req_data_and_make_predictions function.
        app (TestClient): The TestClient fastapi app
    The function creates a mock request and sets the expected return value of the mock_transform_and_predict function.
    It then sends a POST request to the "/infer" endpoint with the mock request data.
    The function asserts that the response status code is 200 and the JSON response matches the expected output.
    Additionally, it checks if the mock_transform_and_predict function was called with the correct arguments.
    """
    # Define what your mock should return
    mock_transform_and_predict.return_value = pd.DataFrame(), {"status": "success", "predictions": []}

    response = app.post("/infer", data=json.dumps(sample_request_data))

    print(response.json())
    assert response.status_code == 200
    assert response.json() == {"status": "success", "predictions": []}
    # You can add more assertions to check if the function was called with the correct arguments
    mock_transform_and_predict.assert_called()
