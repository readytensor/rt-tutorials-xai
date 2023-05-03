import numpy as np
import pandas as pd
import os
from typing import List, Union, Any, Dict, Callable, Tuple
from skopt import gp_minimize
from skopt.space import Real, Categorical, Integer
from skopt.plots import (
    plot_convergence,
    plot_objective,
)
import matplotlib.pyplot as plt

from hyperparameter_tuning.base_tuner import HyperparameterTuner
from utils import save_dataframe_as_csv


class SKOHyperparameterTuner(HyperparameterTuner):
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

        # names of hyperparameters in a list; this is needed in a couple of places
        self.ordered_hps = [hp_obj["name"] for hp_obj in self.hpt_specs["hyperparameters"]]


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
        hpt_space = self.get_hpt_space() 
        default_hps_ordered = [self.default_hps[hp] for hp in self.ordered_hps]
        n_initial_points = max(1, min(self.num_trials // 3, 5))
        n_calls = max(2, self.num_trials)  # gp_minimize needs at least 2 trials, or it throws an error
        minimizer_kwargs = {
            "func": objective_func, # the objective function to minimize
            "dimensions": hpt_space, # the hyperparameter space
            "x0": default_hps_ordered,
            "acq_func":"EI", # the acquisition function
            "n_initial_points":n_initial_points, # Number of evaluations of `func` with initialization points before approximating it with base_estimator
            "n_calls":n_calls, # Number of calls to `func`,
            "verbose": True
        }
        return gp_minimize, minimizer_kwargs


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
                param_grid.append(space_constructor(hp_obj['categorical_vals'], name=hp_obj['name']))
            else:
                param_grid.append(space_constructor(hp_obj['range_low'], hp_obj['range_high'], name=hp_obj['name']))

        return param_grid



    def extract_hyperparameters_from_trial(self, trial: Any) -> Dict[str, Any]:
        """
        Extract the hyperparameters from trial object to pass to the model training function.

        Args:
            trial (Any): Dictionary containing the hyperparameters.

        Returns:
            Dict[str, Any]: Dictionary containing the properly formatted hyperparameters.
        """
        # with scikit-optimize, the trial is a list of values for each hyperparameter, in the
        # same order as defined in self.ordered_hps
        # so the formatting conversion has to be:
        #       [hp_1_val, hp_2_val, ... ] => {"hp_1_name": hp_1_val, "hp_2_name": hp_2_val, ...}
        hyperparameters = {}
        for hp, val in zip(self.ordered_hps, trial):
            hyperparameters[hp] = val
        return hyperparameters


    def save_hpt_summary_results(self,optimizer_result):
        """Save the hyperparameter tuning results to a file.

        Args:
            optimizer_result (Any): The result object returned by the optimizer function.
        """
        # # save trial results
        hpt_results_df = pd.concat([
            pd.DataFrame(optimizer_result.x_iters),
            pd.Series(optimizer_result.func_vals),
        ], axis=1)
        hpt_results_df.columns = self.ordered_hps + ["metric_value"]
        hpt_results_df.insert(0, "trial_num", 1+np.arange(hpt_results_df.shape[0]))
        save_dataframe_as_csv(hpt_results_df, os.path.join(self.hpt_results_dir_path, "hpt_results.csv"))

        # save convergence plot
        plot_convergence(optimizer_result)
        figure = plt.gcf() # get current figure
        figure.tight_layout()
        figure.set_size_inches(10, 6)
        plt.savefig(os.path.join(self.hpt_results_dir_path, "convergence_plot.png"))

        # Partial dependency plots
        plot_objective(result=optimizer_result, plot_dims=self.ordered_hps)
        figure = plt.gcf() # get current figure
        figure.tight_layout()
        fig_ax_wd = 2 + len(self.ordered_hps) * 3  # 3 inches per dimension to make it spacious enough + 2 inches fixed
        fig_ax_ht= 1 + len(self.ordered_hps) * 3  # 3 inches per dimension to make it spacious enough  + 1 inch fixed
        figure.set_size_inches(fig_ax_wd, fig_ax_ht)
        plt.savefig(os.path.join(self.hpt_results_dir_path, "partial_dependence_plot.png"))

        
