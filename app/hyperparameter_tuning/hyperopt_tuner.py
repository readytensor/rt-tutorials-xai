import numpy as np
import pandas as pd
import os
from typing import Callable, Any, Dict, Tuple
from hyperopt import fmin, hp, tpe, Trials
import matplotlib.pyplot as plt

from hyperparameter_tuning.base_tuner import HyperparameterTuner
from utils import save_dataframe_as_csv


class HyperOptHyperparameterTuner(HyperparameterTuner):
    """HyperOpt hyperparameter tuner class.

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

        # Trials captures the search information
        self.trials = Trials()

    
    
    def get_minimizer_func_and_kwargs(self, objective_func: Callable) -> Tuple[Callable, Dict[str, Any]]:
        """Get the minimizer function and its keyword arguments.

        Args:
            objective_func (Callable): Objective function to minimize.

        Returns:
            Tuple[Callable, Dict[str, Any]]: Tuple containing the minimizer function and its keyword arguments.
        """
        hpt_space = self._get_hpt_space() 
        n_calls = max(2, self.num_trials)  # at least 2 trials, otherwise why bother?
        
        minimizer_kwargs = {
            "fn": objective_func, # the objective function to minimize
            "space": hpt_space, # the hyperparameter space
            "algo":tpe.suggest,
            "max_evals": n_calls, # the number of evaluations of the objective function
            "rstate": np.random.default_rng(0), # the random state object to use
            "trials":self.trials,
        }
        return fmin, minimizer_kwargs


    def _get_hpt_space(self) -> Dict[str, Any]:
        """Get the hyperparameter tuning search space.

        Returns:
            Dict[str, Any]: Dictionary of hyperparameter space objects.
        """
        param_grid = {}
        search_types = {
            "int": {
                "uniform": hp.quniform,
                "log-uniform": hp.qloguniform,
            },
            "real": {
                "uniform": hp.uniform,
                "log-uniform": hp.loguniform,
            },
        }
        for hp_obj in self.hpt_specs["hyperparameters"]:
            hp_val_type = hp_obj["type"]
            search_type = hp_obj["search_type"]
            if hp_val_type in search_types and search_type in search_types[hp_val_type]:
                if hp_val_type == "int":
                    val = search_types[hp_val_type][search_type](
                        hp_obj["name"], hp_obj["range_low"], hp_obj["range_high"], 1
                    )
                else:
                    val = search_types[hp_val_type][search_type](
                        hp_obj["name"], hp_obj["range_low"], hp_obj["range_high"]
                    )
            elif hp_val_type == "categorical":
                val = hp.choice(hp_obj["name"], hp_obj["categorical_vals"])
            else:
                raise ValueError(
                    f"Error creating Hyper-Param Grid. "
                    f"Undefined value type: {hp_obj['type']} or search_type: {hp_obj['search_type']}. "
                    "Verify hpt_config.json file."
                )
            param_grid.update({hp_obj["name"]: val})

        return param_grid



    def save_hpt_summary_results(self, optimizer_result: Any) -> None:
        """Save the hyperparameter tuning results to a file.
        
        Args:
            optimizer_result (Any): The result object returned by the optimizer function.
        """
        # we dont need to use the optimizer_result argument for hyperopt.
        # The tuning results are stored in self.trials

        # save trial results
        hpt_results_df = pd.concat([
            pd.DataFrame(self.trials.vals),
            pd.DataFrame(self.trials.results)],
            axis=1,
        ).sort_values(by='loss', ascending=False).reset_index(drop=True)
        hpt_results_df.insert(0, "trial_num", 1+np.arange(hpt_results_df.shape[0]))
        save_dataframe_as_csv(hpt_results_df, os.path.join(self.hpt_results_dir_path, "hpt_results.csv"))

         # save convergence plot
        figsize=(10, 6)
        if self.is_minimize:
            hpt_results_df['loss'].sort_values().reset_index(drop=True).plot(figsize=figsize)
        else:
            hpt_results_df['loss'].\
                sort_values(ascending=False).reset_index(drop=True).plot(figsize=figsize)
        plt.title('Convergence plot')
        plt.xlabel('Iteration')
        plt.ylabel('Metric Value')
        plt.savefig(os.path.join(self.hpt_results_dir_path, "convergence_plot.png"))


