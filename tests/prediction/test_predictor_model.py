import pytest
import os
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from prediction.predictor_model import (
    Classifier,
    train_predictor_model,
    predict_with_model,
    save_predictor_model,
    load_predictor_model,
    evaluate_predictor_model
)

# Define the hyperparameters fixture
@pytest.fixture
def hyperparameters():
    return {
        "n_estimators": 10,
        "min_samples_split": 4,
        "min_samples_leaf": 2,
    }

    
@pytest.fixture
def classifier(hyperparameters):
    """ Define the classifier fixture"""
    return Classifier(**hyperparameters)


@pytest.fixture
def synthetic_data():
    """ Define the synthetic dataset fixture"""
    X, y = make_classification(n_samples=100, n_features=5, random_state=42)
    train_X, train_y = X[:80], y[:80]
    test_X, test_y = X[80:], y[80:]
    return train_X, train_y, test_X, test_y


def test_build_model(classifier, hyperparameters):
    """ Test if the build_model method creates a model with the specified hyperparameters. """
    model = classifier.build_model()
    assert isinstance(model, classifier.model.__class__)
    for param, value in hyperparameters.items():
        assert getattr(model, param) == value


def test_build_model_without_hyperparameters():
    """
    Test if the build_model method creates a model with default hyperparameters when none are provided.
    """
    default_classifier = Classifier()
    model = default_classifier.build_model()

    assert isinstance(model, default_classifier.model.__class__)

    # Check if the model has default hyperparameters
    default_hyperparameters = {
        "n_estimators": 200,
        "min_samples_split": 8,
        "min_samples_leaf": 4,
    }

    for param, value in default_hyperparameters.items():
        assert getattr(model, param) == value


def test_fit_predict_evaluate(classifier, synthetic_data):
    """
    Test if the fit method trains the model correctly and if the predict and evaluate methods work as expected.
    """
    train_X, train_y, test_X, test_y = synthetic_data
    classifier.fit(train_X, train_y)
    predictions = classifier.predict(test_X)
    assert predictions.shape == test_y.shape
    assert np.array_equal(predictions, predictions.astype(bool))

    proba_predictions = classifier.predict_proba(test_X)
    assert proba_predictions.shape == (test_y.shape[0], 2)

    accuracy = classifier.evaluate(test_X, test_y)
    assert isinstance(accuracy, float)
    assert 0 <= accuracy <= 1


def test_save_load(tmpdir_factory, classifier, synthetic_data, hyperparameters):
    """
    Test if the save and load methods work correctly and if the loaded model has the same hyperparameters
    and predictions as the original.
    """
    # Convert the LocalPath object to a string
    tmpdir_str = str(tmpdir_factory.mktemp("data"))

    # Specify the file path
    model_file_path = os.path.join(tmpdir_str, 'model.joblib')

    train_X, train_y, test_X, test_y = synthetic_data
    classifier.fit(train_X, train_y)

    # Save the model
    classifier.save(model_file_path)

    # Load the model
    loaded_clf = Classifier.load(model_file_path)
    
    # Check the loaded model has the same hyperparameters as the original classifier
    for param, value in hyperparameters.items():
        assert getattr(loaded_clf.model, param) == value

    # Test predictions
    predictions = loaded_clf.predict(test_X)
    assert np.array_equal(predictions, classifier.predict(test_X))

    proba_predictions = loaded_clf.predict_proba(test_X)
    assert np.array_equal(proba_predictions, classifier.predict_proba(test_X))

    # Test evaluation
    accuracy = loaded_clf.evaluate(test_X, test_y)
    assert accuracy == classifier.evaluate(test_X, test_y)


def test_accuracy_compared_to_logistic_regression(classifier, synthetic_data):
    """
    Test if the accuracy of the classifier is close enough to the accuracy of a baseline model like logistic regression.
    """
    train_X, train_y, test_X, test_y = synthetic_data

    # Fit and evaluate the classifier
    classifier.fit(train_X, train_y)
    classifier_accuracy = classifier.evaluate(test_X, test_y)

    # Fit and evaluate the logistic regression model
    baseline_model = LogisticRegression()
    baseline_model.fit(train_X, train_y)
    baseline_accuracy = baseline_model.score(test_X, test_y)

    # Set an acceptable difference in accuracy
    accuracy_threshold = -0.05

    # Check if the classifier's accuracy is close enough to the logistic regression accuracy
    assert classifier_accuracy - baseline_accuracy > accuracy_threshold


def test_train_predictor_model(synthetic_data, hyperparameters):
    """
    Test that the 'train_predictor_model' function returns a Classifier instance with correct hyperparameters.
    """
    train_X, train_y, _, _ = synthetic_data
    classifier = train_predictor_model(train_X, train_y, hyperparameters)
    
    assert isinstance(classifier, Classifier)
    for param, value in hyperparameters.items():
        assert getattr(classifier.model, param) == value


def test_predict_with_model(synthetic_data, hyperparameters):
    """
    Test that the 'predict_with_model' function returns predictions of correct size and type.
    """
    train_X, train_y, test_X, _ = synthetic_data
    classifier = train_predictor_model(train_X, train_y, hyperparameters)
    predictions = predict_with_model(classifier, test_X)
    
    assert isinstance(predictions, np.ndarray)
    assert predictions.shape[0] == test_X.shape[0]


def test_save_predictor_model(tmpdir_factory, classifier):
    """
    Test that the 'save_predictor_model' function correctly saves a Classifier instance to disk.
    """
    tmpdir_str = str(tmpdir_factory.mktemp("data"))
    model_file_path = os.path.join(tmpdir_str, 'model.joblib')
    
    save_predictor_model(classifier, model_file_path)
    assert os.path.exists(model_file_path)


def test_load_predictor_model(tmpdir_factory, classifier, hyperparameters):
    """
    Test that the 'load_predictor_model' function correctly loads a Classifier instance from disk and that the loaded instance has the correct hyperparameters.
    """
    tmpdir_str = str(tmpdir_factory.mktemp("data"))
    model_file_path = os.path.join(tmpdir_str, 'model.joblib')
    classifier.save(model_file_path)

    loaded_clf = load_predictor_model(model_file_path)
    assert isinstance(loaded_clf, Classifier)
    for param, value in hyperparameters.items():
        assert getattr(loaded_clf.model, param) == value


def test_evaluate_predictor_model(synthetic_data, hyperparameters):
    """
    Test that the 'evaluate_predictor_model' function returns an accuracy score of correct type and within valid range.
    """
    train_X, train_y, test_X, test_y = synthetic_data
    classifier = train_predictor_model(train_X, train_y, hyperparameters)
    accuracy = evaluate_predictor_model(classifier, test_X, test_y)
    
    assert isinstance(accuracy, float)
    assert 0 <= accuracy <= 1
