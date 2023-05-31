from fastapi.testclient import TestClient
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


@pytest.fixture
def sample_explanation_response_data():
    # Define a fixture for expected response
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


def test_infer_endpoint_integration(app, sample_request_data, sample_response_data):
    """
    End-to-end integration test for the /infer endpoint of the FastAPI application.

    This test uses a TestClient from FastAPI to make a POST request to the /infer endpoint,
    and verifies that the response matches expectations.

    A ModelResources instance is created with test-specific paths using the test_model_resources fixture,
    and the application's dependency on ModelResources is overridden to use this instance for the test.

    The function sends a POST request to the "/infer" endpoint with the test_sample_request_data
    using a TestClient from FastAPI.
    It then asserts that the response keys match the expected response keys, and compares specific
    values in the returned response_data with the sample_response_data.
    Finally, it resets the dependency_overrides after the test.

    Args:
        app (TestClient): The test app.
        sample_request_data (dict): The fixture for test request data.
        sample_response_data (dict): The fixture for expected response data.
    Returns:
        None
    """
    response = app.post("/infer", json=sample_request_data)
    response_data = response.json()

    # assertions
    assert set(response_data.keys()) == set(response.json().keys())
    assert response_data["predictions"][0]["sampleId"] == \
        sample_response_data["predictions"][0]["sampleId"]
    assert response_data["predictions"][0]["predictedClass"] == \
        sample_response_data["predictions"][0]["predictedClass"]


def test_explain_endpoint_integration(
        app, sample_request_data, sample_explanation_response_data):
    """
    End-to-end integration test for the /explain endpoint of the FastAPI application.

    This test uses a TestClient from FastAPI to make a POST request to the /explain endpoint,
    and verifies that the response matches expectations.

    A ModelResources instance is created with test-specific paths using the test_model_resources fixture,
    and the application's dependency on ModelResources is overridden to use this instance for the test.

    The function sends a POST request to the "/explain" endpoint with the test_request_data
    using a TestClient from FastAPI.
    It then asserts that the response keys match the expected response keys, and compares specific
    values in the explanation_response_data with the sample_explanation_response_data.
    Finally, it resets the dependency_overrides after the test.

    Args:
        test_model_resources (ModelResources): The test ModelResources object.
        test_request_data (dict): The fixture for test request data.
        sample_explanation_response_data (dict): The fixture for expected explanation response data.
    Returns:
        None
    """
    response = app.post("/explain", json=sample_request_data)
    explanation_response_data = response.json()

    # assertions
    assert set(explanation_response_data.keys()) == set(sample_explanation_response_data.keys())
    assert explanation_response_data["predictions"][0]["sampleId"] == sample_explanation_response_data["predictions"][0]["sampleId"]
    assert explanation_response_data["predictions"][0]["predictedClass"] == sample_explanation_response_data["predictions"][0]["predictedClass"]
    
    # baseline assertions
    assert explanation_response_data["predictions"][0]["explanation"].get("baseline") is not None
    baseline = explanation_response_data["predictions"][0]["explanation"]["baseline"]
    assert len(baseline) == 2
    assert round(sum(baseline), 4) == 1.0000

    # explanation assertions
    assert explanation_response_data["predictions"][0]["explanation"].get("featureScores") is not None
    feature_scores = explanation_response_data["predictions"][0]["explanation"]["featureScores"]
    assert len(feature_scores) == 11
