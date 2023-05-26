import numpy as np
import pandas as pd
import math
from typing import List, Union, Any, Dict, Callable
from skopt import gp_minimize
from skopt.space import Real, Categorical, Integer

from config import paths
from utils import read_json_as_dict, save_dataframe_as_csv
from prediction.predictor_model import (
    train_predictor_model,
    evaluate_predictor_model
)



class SKOHyperparameterTuner():
    """Scikit-Optimize hyperparameter tuner class.

    Args:
        default_hps (Dict[str, Any]): Dictionary of default hyperparameter values.
        hpt_specs (Dict[str, Any]): Dictionary of hyperparameter tuning specs.
        hpt_results_file_path (str): Path to the hyperparameter tuning results file.
        is_minimize (bool, optional): Whether the metric should be minimized. Defaults to True.
    """
    def __init__(
        self,
        default_hyperparameters: Dict[str, Any],
        hpt_specs: Dict[str, Any],
        hpt_results_file_path: str,
        is_minimize: bool = True,
    ):
        """Initializes an instance of the hyperparameter tuner.

        Args:
            default_hyperparameters: Dictionary of default hyperparameter values.
            hpt_specs: Dictionary of hyperparameter tuning specs.
            hpt_results_file_path: Path to the hyperparameter tuning results file.
            is_minimize:  Whether the metric should be minimized or maximized.
                Defaults to True.
        """
        self.default_hyperparameters = default_hyperparameters
        self.hpt_specs = hpt_specs
        self.hpt_results_file_path = hpt_results_file_path
        self.is_minimize = is_minimize
        self.num_trials = hpt_specs.get("num_trials", 20)
        assert self.num_trials >= 2, "Scikit-Optimize minimizer needs at least 2 trials"

        # names of hyperparameters in a list
        self.hyperparameter_names = [hp_obj["name"] for hp_obj in self.hpt_specs["hyperparameters"]]
        self.default_hyperparameter_vals = [self.default_hyperparameters[hp] for hp in self.hyperparameter_names]
        self.hpt_space = self.get_hpt_space()


    def _get_objective_func(
        self,
        train_X: Union[pd.DataFrame, np.ndarray], 
        train_y: Union[pd.Series, np.ndarray], 
        valid_X: Union[pd.DataFrame, np.ndarray], 
        valid_y: Union[pd.Series, np.ndarray]
    ) -> Callable:
        """Gets the objective function for hyperparameter tuning.

        Args:
            train_X: Training data features.
            train_y: Training data labels.
            valid_X: Validation data features.
            valid_y: Validation data labels.

        Returns:
            A callable objective function for hyperparameter tuning.
        """

        def objective_func(trial):
            """Build a model from this hyper parameter permutation and evaluate its performance"""
            # convert list of HP values into a dictionary of name:val pairs
            hyperparameters = dict(zip(self.hyperparameter_names, trial))
            # train model
            classifier = train_predictor_model(train_X, train_y, hyperparameters)
            # evaluate the model
            score = round(evaluate_predictor_model(classifier, valid_X, valid_y), 6)
            if np.isnan(score) or math.isinf(score):
                # sometimes loss becomes inf/na, so use a large "bad" value
                score = 1.0e6 if self.is_minimize else -1.0e6
            # If this is a maximization metric then return negative of it
            return score if self.is_minimize else -score
        
        return objective_func
    

    def get_hpt_space(self) -> List[Union[Categorical, Integer, Real]]:
        """Get the hyperparameter tuning search space.

        Returns:
            List[Union[Categorical, Integer, Real]]: List of hyperparameter space objects.
        """
        param_grid = []
        space_map = {
            ('categorical', None): Categorical,
            ('int', 'uniform'): lambda low, high, name: Integer(low, high, prior='uniform', name=name),
            ('int', 'log-uniform'): lambda low, high, name: Integer(low, high, prior='log-uniform', name=name),
            ('real', 'uniform'): lambda low, high, name: Real(low, high, prior='uniform', name=name),
            ('real', 'log-uniform'): lambda low, high, name: Real(low, high, prior='log-uniform', name=name)
        }

        for hp_obj in self.hpt_specs["hyperparameters"]:
            method_key = (hp_obj["type"], hp_obj.get("search_type"))
            space_constructor = space_map.get(method_key)

            if space_constructor is None:
                raise ValueError(f"Error creating Hyper-Param Grid. \
                    Undefined value type: {hp_obj['type']} or search_type: {hp_obj['search_type']}. \
                    Verify hpt_config.json file.")
            
            if hp_obj["type"] == 'categorical':
                param_grid.append(space_constructor(hp_obj['categories'], name=hp_obj['name']))
            else:
                param_grid.append(space_constructor(hp_obj['range_low'], hp_obj['range_high'], name=hp_obj['name']))

        return param_grid


    def run_hyperparameter_tuning(
        self,
        train_X: Union[pd.DataFrame, np.ndarray],
        train_y: Union[pd.Series, np.ndarray],
        valid_X: Union[pd.DataFrame, np.ndarray],
        valid_y: Union[pd.Series, np.ndarray]
    ) -> Dict[str, Any]:
        """Runs the hyperparameter tuning process.

        Args:
            train_X: Training data features.
            train_y: Training data labels.
            valid_X: Validation data features.
            valid_y: Validation data labels.

        Returns:
            A dictionary containing the best model name, hyperparameters, and score.
        """
        # Use 1/3 of the trials to explore the space initially, but at most 5 trials
        n_initial_points = max(1, min(self.num_trials // 3, 5))
        objective_func = self._get_objective_func(train_X, train_y, valid_X, valid_y)
        optimizer_results = gp_minimize(
            func=objective_func, # the objective function to minimize
            dimensions=self.hpt_space, # the hyperparameter space
            x0= self.default_hyperparameter_vals,
            acq_func="EI", # the acquisition function
            n_initial_points=n_initial_points, # Number of evaluations of `func` with initialization points before approximating it with base_estimator
            n_calls=self.num_trials, # Number of calls to `func`,
            random_state=0,
            verbose= True
        )
        self.save_hpt_summary_results(optimizer_results)
        return self.get_best_hyperparameters(optimizer_results)
    

    def get_best_hyperparameters(self, optimizer_results: Any) -> Dict[str, Any]:
        """Gets the best hyperparameters from the optimization results.

        Args:
            optimizer_results: The result object returned by the optimizer function.

        Returns:
            A dictionary containing the best hyperparameters.
        """
        best_idx = np.argmin(optimizer_results.func_vals)
        best_hyperparameter_vals = optimizer_results.x_iters[best_idx]
        best_hyperparameters = dict(zip(self.hyperparameter_names, best_hyperparameter_vals))
        return best_hyperparameters


    def save_hpt_summary_results(self, optimizer_result: Any):
        """Saves the hyperparameter tuning results to a file.

        Args:
            optimizer_result: The result object returned by the optimizer function.
        """
        # # save trial results
        hpt_results_df = pd.concat([
            pd.DataFrame(optimizer_result.x_iters),
            pd.Series(optimizer_result.func_vals),
        ], axis=1)
        hpt_results_df.columns = self.hyperparameter_names + ["metric_value"]        
        hpt_results_df.insert(0, "trial_num", 1+np.arange(hpt_results_df.shape[0]))
        hpt_results_df.sort_values(by="metric_value", inplace=True, ignore_index=True)
        if self.is_minimize is False:
            hpt_results_df["metric_value"] = -hpt_results_df["metric_value"]
        save_dataframe_as_csv(hpt_results_df, self.hpt_results_file_path)


def tune_hyperparameters(
    train_X: Union[pd.DataFrame, np.ndarray],
    train_y: Union[pd.Series, np.ndarray],
    valid_X: Union[pd.DataFrame, np.ndarray],
    valid_y: Union[pd.Series, np.ndarray],
    hpt_results_file_path: str,
    is_minimize: bool = True,
    default_hyperparameters_file_path: str = paths.DEFAULT_HYPERPARAMETERS_FILE_PATH,
    hpt_specs_file_path: str = paths.HPT_CONFIG_FILE_PATH
) -> Dict[str, Any]:
    """
    Tune hyperparameters using Scikit-Optimize (SKO) hyperparameter tuner.

    This function creates an instance of the SKOHyperparameterTuner with the
    provided hyperparameters and tuning specifications, then runs the hyperparameter
    tuning process and returns the best hyperparameters.

    Args:
        train_X (Union[pd.DataFrame, np.ndarray]): Training data features.
        train_y (Union[pd.Series, np.ndarray]): Training data labels.
        valid_X (Union[pd.DataFrame, np.ndarray]): Validation data features.
        valid_y (Union[pd.Series, np.ndarray]): Validation data labels.
        hpt_results_file_path (str): Path to the hyperparameter tuning results file.
        is_minimize (bool, optional): Whether the metric should be minimized. Defaults to True.
        default_hyperparameters_file_path (str, optional): Path to the json file with default hyperparameter values. 
            Defaults to the path defined in the paths.py file.
        hpt_specs_file_path (str, optional): Path to the json file with hyperparameter tuning specs.
            Defaults to the path defined in the paths.py file.

    Returns:
        Dict[str, Any]: Dictionary containing the best hyperparameters.
    """
    default_hyperparameters = read_json_as_dict(default_hyperparameters_file_path)
    hpt_specs = read_json_as_dict(hpt_specs_file_path)
    hyperparameter_tuner = SKOHyperparameterTuner(
        default_hyperparameters=default_hyperparameters,
        hpt_specs=hpt_specs,
        hpt_results_file_path=hpt_results_file_path,
        is_minimize=is_minimize)
    best_hyperparams = hyperparameter_tuner.run_hyperparameter_tuning(
        train_X, train_y, valid_X, valid_y)
    return best_hyperparams
