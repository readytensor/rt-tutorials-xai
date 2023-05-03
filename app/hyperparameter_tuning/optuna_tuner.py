import os
from typing import  Any, Dict, Union, Callable, Tuple
import optuna
import matplotlib.pyplot as plt

from hyperparameter_tuning.base_tuner import HyperparameterTuner

from utils import save_dataframe_as_csv

class OptunaHyperparameterTuner(HyperparameterTuner):
    """Scikit-Optimize hyperparameter tuner class.

    Args:
        default_hps (Dict[str, Any]): Dictionary of default hyperparameter values.
        hpt_specs (Dict[str, Any]): Dictionary of hyperparameter tuning specs.
        hpt_results_dir_path (str): Path to the hyperparameter tuning results directory.
        best_hp_file_path (str): Path to the file to save the best hyperparameters.
        is_minimize (bool, optional): Whether the metric should be minimized. Defaults to True.
    """
    def __init__(
        self,
        default_hps: Dict[str, Any],
        hpt_specs: Dict[str, Any],
        hpt_results_dir_path: str,
        best_hp_file_path: str,
        is_minimize: bool = True,
    ):
        super().__init__(default_hps, hpt_specs, hpt_results_dir_path, best_hp_file_path, is_minimize)
        self.study = optuna.create_study(
            direction="minimize", #always minimize, because base class handles the direction in the objective function
        )

    def get_minimizer_func_and_kwargs(self, \
            objective_func: Callable) -> Tuple[Callable, Dict[str, Any]]:
        """
        Get the minimizer function and its keyword arguments.

        Args:
            objective_func (Callable): Objective function to minimize.

        Returns:
            Tuple[Callable, Dict[str, Any]]: Tuple containing the minimizer function and its keyword arguments.
        """
        minimizer_func = self.study.optimize
        minimizer_kwargs = {
            "func": objective_func, # the objective function to minimize
            "n_trials": self.num_trials, # the number of trials
            "n_jobs": 1, # the number of parallel jobs, change this if you have multiple cores
            "gc_after_trial": True, # Flag to determine whether to automatically run garbage collection after each trial.
            "show_progress_bar": True, #Flag to show progress bars or not
        }
        return minimizer_func, minimizer_kwargs
  

    def extract_hyperparameters_from_trial(self, trial: Any) -> Dict[str, Any]:
        """
        Extract the hyperparameters from trial object to pass to the model training function.

        Args:
            trial (Any): Dictionary containing the hyperparameters.

        Returns:
            Dict[str, Any]: Dictionary containing the properly formatted hyperparameters.
        """        
        hyperparameters = {}
        trial_suggest_methods = {
            ('categorical', None): trial.suggest_categorical,
            ('int', 'uniform'): trial.suggest_int,
            ('int', 'log-uniform'): lambda name, low, high: trial.suggest_int(name, low, high, log=True),
            ('real', 'uniform'): trial.suggest_float,
            ('real', 'log-uniform'): lambda name, low, high: trial.suggest_float(name, low, high, log=True)
        }
        for hp_obj in self.hpt_specs["hyperparameters"]:
            method_key = (hp_obj["type"], hp_obj.get("search_type"))
            suggest_method = trial_suggest_methods.get(method_key)

            if suggest_method is None:
                raise ValueError(f"Error creating Hyper-Param Grid. \
                    Undefined value type: {hp_obj['type']} or search_type: {hp_obj['search_type']}. \
                    Verify hpt_config.json file.")
            
            if hp_obj["type"] == 'categorical':
                hyperparameters[hp_obj['name']] = suggest_method(
                    hp_obj['name'],
                    hp_obj['categorical_vals']
                )
            else:
                hyperparameters[hp_obj['name']] = suggest_method(
                    hp_obj['name'],
                    hp_obj['range_low'],
                    hp_obj['range_high']
                )
        return hyperparameters


    def save_hpt_summary_results(self, optimizer_result):
        """Save the hyperparameter tuning results to a file.
        
        Args:
            optimizer_result (Any): The result object returned by the optimizer function.
        """

        # save trial results
        hpt_results_df = self.study.trials_dataframe()
        save_dataframe_as_csv(hpt_results_df, os.path.join(self.hpt_results_dir_path, "hpt_results.csv"))

        # save convergence plot
        figsize=(10, 6)
        if self.is_minimize:
            hpt_results_df['value'].sort_values().reset_index(drop=True).plot(figsize=figsize)
        else:
            hpt_results_df['value'].sort_values(ascending=False).reset_index(drop=True).plot(figsize=figsize)
        plt.title('Convergence plot')
        plt.xlabel('Iteration')
        plt.ylabel('Metric Value')
        plt.savefig(os.path.join(self.hpt_results_dir_path, "convergence_plot.png"))