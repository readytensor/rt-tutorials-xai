import pandas as pd
from pandas import DataFrame, Series
import numpy as np
from typing import Any, List, Tuple
import pytest
from pathlib import Path
from py.path import local as LocalPath

from src.xai.explainer import (
    ShapClassificationExplainer, 
    fit_and_save_explainer, 
    load_explainer,
    get_explanations_from_explainer)


def test_fit_explainer(transformed_train_inputs: DataFrame) -> None:
    """
    Test fitting of the explainer.

    Args:
        transformed_train_inputs (DataFrame): Transformed train inputs.
    """
    explainer = ShapClassificationExplainer()
    explainer.fit(transformed_train_inputs)
    assert explainer._explainer_data.shape == transformed_train_inputs.shape


def test_build_explainer(transformed_train_inputs: DataFrame, predictor: Any) -> None:
    """
    Test building of the explainer.

    Args:
        transformed_train_inputs (DataFrame): Transformed train inputs.
        predictor (Any): A predictor model object.
    """
    explainer = ShapClassificationExplainer()
    explainer.fit(transformed_train_inputs)
    class_names = ['class_0', 'class_1']  # Assuming binary classification
    shap_explainer = explainer._build_explainer(predictor, class_names)
    assert shap_explainer is not None


def test_get_explanations(
        transformed_test_inputs: DataFrame, predictor: Any,
        class_names: List[str]) -> None:
    """
    Test getting explanations.

    Args:
        transformed_test_inputs (DataFrame): Transformed test inputs.
        predictor (Any): A predictor model object.
        class_names (List[str]): List of class names.

    """
    explainer = ShapClassificationExplainer()
    explainer.fit(transformed_test_inputs)
    explanations = explainer.get_explanations(transformed_test_inputs, predictor, class_names)
    assert explanations is not None
    assert "explanation_method" in explanations
    assert "explanations" in explanations


def test_save_and_load_explainer(tmpdir: LocalPath, transformed_train_inputs: DataFrame) -> None:
    """
    Test saving and loading of the explainer.

    Args:
        tmpdir (LocalPath): Temporary directory path provided by pytest.
        transformed_train_inputs (DataFrame): Transformed train inputs.

    """
    explainer = ShapClassificationExplainer(max_local_explanations=10)
    explainer.fit(transformed_train_inputs)
    file_path = tmpdir.join('explainer.pkl')
    explainer.save(file_path)
    loaded_explainer = ShapClassificationExplainer.load(file_path)
    assert loaded_explainer is not None
    assert loaded_explainer._explainer_data.shape == transformed_train_inputs.shape
    assert loaded_explainer.max_local_explanations == 10


def test_fit_and_save_explainer(
        transformed_train_inputs: DataFrame,
        explainer_config_file_path: str, explainer_file_path: str) -> None:
    """
    Test fitting and saving of the explainer.

    Args:
        transformed_train_inputs (DataFrame): Transformed train inputs.
        explainer_config_file_path (str): Path to the explainer configuration file.
        explainer_file_path (str): Path where the explainer is to be saved.

    """
    explainer = fit_and_save_explainer(
        transformed_train_inputs, explainer_config_file_path, explainer_file_path)
    assert Path(explainer_file_path).is_file()
    assert explainer is not None
    assert explainer._explainer_data.shape == transformed_train_inputs.shape


def test_load_explainer(
        transformed_train_inputs: DataFrame,
        explainer_config_file_path: str, explainer_file_path: str) -> None:
    """
    Test loading of the explainer.

    Args:
        transformed_train_inputs (DataFrame): Transformed train inputs.
        explainer_config_file_path (str): Path to the explainer configuration file.
        explainer_file_path (str): Path where the explainer is to be saved.

    """
    _ = fit_and_save_explainer(
        transformed_train_inputs, explainer_config_file_path, explainer_file_path)
    loaded_explainer = load_explainer(explainer_file_path)
    assert loaded_explainer is not None
    assert loaded_explainer._explainer_data.shape == transformed_train_inputs.shape


def test_get_explanations_from_explainer(
        transformed_test_inputs: DataFrame, explainer_config_file_path: str, explainer_file_path: str,
        predictor: Any, class_names: List[str]) -> None:
    """
    Test the test_get_explanations_from_explainer function.

    Args:
        transformed_test_inputs (DataFrame): Transformed test inputs.
        explainer_config_file_path (str): Path to the explainer configuration file.
        explainer_file_path (str): Path where the explainer is to be saved.
        predictor (Any): A predictor model object.
        class_names (List[str]): List of class names.

    """
    explainer = fit_and_save_explainer(
        transformed_test_inputs, explainer_config_file_path, explainer_file_path)
    explanations = get_explanations_from_explainer(
        transformed_test_inputs, explainer, predictor, class_names
    )
    assert explanations is not None
    assert "explanation_method" in explanations
    assert "explanations" in explanations
    assert isinstance(explanations["explanations"], list)
    for explanation in explanations["explanations"]:
        assert isinstance(explanation, dict)
        assert "baseline" in explanation
        assert "featureScores" in explanation


def test_explanations_from_loaded_explainer(
        transformed_train_inputs: DataFrame, explainer_config_file_path: str, explainer_file_path: str,
        predictor: Any, transformed_test_inputs: DataFrame, class_names: List[str]) -> None:
    """
    Test loading of the explainer and getting explanations.

    Args:
        transformed_train_inputs (DataFrame): Transformed train inputs.
        explainer_config_file_path (str): Path to the explainer configuration file.
        explainer_file_path (str): Path where the explainer is to be saved.
        predictor (Any): A predictor model object.
        transformed_test_inputs (DataFrame): Transformed test inputs.
        class_names (List[str]): List of class names.

    """
    fit_and_save_explainer(
        transformed_train_inputs, explainer_config_file_path, explainer_file_path)
    loaded_explainer = load_explainer(explainer_file_path)
    assert loaded_explainer._explainer_data.shape == transformed_train_inputs.shape

    explanations = get_explanations_from_explainer(
        transformed_test_inputs, loaded_explainer, predictor, class_names
    )
    assert explanations is not None
