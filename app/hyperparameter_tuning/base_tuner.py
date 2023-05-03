from abc import ABC, abstractmethod
from typing import Tuple, Union, Any, Dict, Callable
import random
import os
import string
import numpy as np
import pandas as pd
import math

from utils import read_json_as_dict, save_json
from algorithm.classifier import (
    train_classifier_model,
    evaluate_classifier
)

class HyperparameterTuner(ABC):
    def __init__(
        self,
        default_hps: Dict[str, Any],
        hpt_specs: Dict[str, Any],
        hpt_results_dir_path: str,
        best_hp_file_path: str,
        is_minimize: bool = True,
    ):
        self.default_hps = default_hps
        self.hpt_specs = hpt_specs
        self.hpt_results_dir_path = hpt_results_dir_path
        self.best_hp_file_path = best_hp_file_path
        self.is_minimize = is_minimize
        self.num_trials = hpt_specs.get("num_trials", 20)


    def run_hyperparameter_tuning(
        self,
        train_X: Union[pd.DataFrame, np.ndarray],
        train_y: Union[pd.Series, np.ndarray],
        valid_X: Union[pd.DataFrame, np.ndarray],
        valid_y: Union[pd.Series, np.ndarray],
    ) -> Dict[str, Any]:
        """Run the hyperparameter tuning process.

        Args:
            train_X (Union[pd.DataFrame, np.ndarray]): Training data features.
            train_y (Union[pd.Series, np.ndarray]): Training data labels.
            valid_X (Union[pd.DataFrame, np.ndarray]): Validation data features.
            valid_y (Union[pd.Series, np.ndarray]): Validation data labels.

        Returns:
            Dict[str, Any]: Dictionary containing the best model name, hyperparameters, and score.
        """

        print("Running HPT ...")
        
        # clear the results directory
        self.clear_hpt_results_dir()
        
        # define the objective function which will be minimized by the optimizer
        def objective_func(trial):
            """Build a model from this hyper parameter permutation and evaluate its performance"""

            hyperparameters = self.extract_hyperparameters_from_trial(trial)

            # train model
            classifier = train_classifier_model(train_X, train_y, hyperparameters)

            # evaluate the model
            score = np.round(evaluate_classifier(classifier, valid_X, valid_y), 6)
            if np.isnan(score) or math.isinf(score):
                # sometimes loss becomes inf/na, so use a large "bad" value
                score = 1.0e6 if self.is_minimize else -1.0e6

            # create a unique model name for the trial - we add loss into file name
            # so we can later sort by file names, and get the best score file without reading each file
            result_name = self._get_trial_result_name(score)

            # create trial result dict
            result = {
                "model_name": result_name,
                "hyperparameters": hyperparameters,
                "score": score
            }

            # Save trial result to disk with unique filename
            save_json(os.path.join(self.hpt_results_dir_path, result_name + ".json"), result)

            # Save the best model parameters found so far in case the HPO job is killed
            self.save_best_hyperparameters()

            # All optimizers should be set to minimize the metric, so if this is a maximization metric
            # then return negative of it
            return score if self.is_minimize else -score

        # these are specific to the derived classes
        minimizer_func, minimizer_kwargs = self.get_minimizer_func_and_kwargs(objective_func)
        optimizer_result = minimizer_func(**minimizer_kwargs)
        print("Completed HPT")

        # save HPT summary results
        self.save_hpt_summary_results(optimizer_result)

        return self.load_best_hyperparams_results()


    @abstractmethod
    def get_minimizer_func_and_kwargs(self, \
            objective_func: Callable) -> Tuple[Callable, Dict[str, Any]]:
        """
        Return the hyperparameter space to be used for the hyperparameter tuning.
        This method must be implemented by the derived class.

        Args:
            objective_func (Callable): Objective function to minimize.

        Returns:
            Tuple[Callable, Dict[str, Any]]: Tuple containing the minimizer function and its keyword arguments.
        """


    def extract_hyperparameters_from_trial(self, trial: Any) -> Dict[str, Any]:
        """Extract the hyperparameters to be passed to the model training function from trial object.

        Derived classes should overwrite this method if the sampled hyperparameters
        from their hyperparameter space need to be extracted/formatted before being passed to the
        train function.  The train function expects hyperparameters to be a dictionary
        with hyperparameter names and their assigned values as the key-value pairs.

        If the hyperparameters do not need to be formatted, then this method does not
        need to be overwritten.

        Args:
            trial (Any): Dictionary containing the hyperparameters.

        Returns:
            Dict[str, Any]: Dictionary containing the properly formatted hyperparameters.
        """
        hyperparameters = trial
        return hyperparameters


    def clear_hpt_results_dir(self) -> None:
        """Clear the hyperparameter tuning results directory."""
        if os.path.exists(self.hpt_results_dir_path):
            for file in os.listdir(self.hpt_results_dir_path):
                if file != ".gitignore":
                    os.remove(os.path.join(self.hpt_results_dir_path, file))
        else:
            os.makedirs(self.hpt_results_dir_path)


    def save_best_hyperparameters(self) -> None:
        """Save best hyperparameters found yet in the `artifacts` folder.
            Also, formats the data types to be int, float or str values for each hyperparameter.
        """
        best_results = self.load_best_hyperparams_results()
        best_hps = best_results["hyperparameters"]
        if best_results is not None:
            formatted_best_hp = {}
            for hp_obj in self.hpt_specs["hyperparameters"]:
                if hp_obj["type"] == "int":
                    formatted_best_hp[hp_obj["name"]] = int(best_hps[hp_obj["name"]])
                elif hp_obj["type"] == "real":
                    formatted_best_hp[hp_obj["name"]] = float(best_hps[hp_obj["name"]])
                elif hp_obj["type"] == "categorical":
                    formatted_best_hp[hp_obj["name"]] = str(best_hps[hp_obj["name"]])
            save_json(self.best_hp_file_path, formatted_best_hp)


    def load_best_hyperparams_results(self) -> Union[Dict[str, Any], None]:
        """Load the best hyperparameter tuning results found so far.

        Returns:
            Union[Dict[str, Any], None]: Dictionary containing the best model name, hyperparameters,
                                        and score, or None if no results are available.
        """
        results = [ f for f in list(sorted(os.listdir(self.hpt_results_dir_path))) if 'json' in f ]
        if len(results) == 0:
            return None
        if self.is_minimize:
            best_result_name = results[0]   # get minimum value
        else:
            best_result_name = results[-1]   # get maximum value
        best_result_file_path = os.path.join(self.hpt_results_dir_path, best_result_name)
        return read_json_as_dict(best_result_file_path)


    def _get_trial_result_name(self, score) -> str:
        """Generate a unique ID for a hyperparameter tuning trial using the score.
        To avoid duplicated names when scores are tied, we concatenate a randomly generated string.

        Args:
            score (float): The score for the given trial.

        Returns:
            str: result name.
        """
        # Create a set of alphanumeric characters
        characters = string.ascii_letters + string.digits
        # Generate the random string
        uniq_id = ''.join(random.choice(characters) for _ in range(6))
        result_name = f"model_score_{str(score)}_{uniq_id}"
        return result_name


    def save_hpt_summary_results(self, optimizer_result: Any) -> None:
        """
        Save the hyperparameter tuning summary results such as convergence plots,
        all trial results, etc.
        The derived classes can implement this specific to their HPT framework/library/custom code.

        Args:
            optimizer_result (Any): The result object returned by the optimizer function.
        """
        pass