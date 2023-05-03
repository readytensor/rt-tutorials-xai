# from hyperparameter_tuning.hyperopt_tuner import HyperOptHyperparameterTuner as Optimizer
# from hyperparameter_tuning.optuna_tuner import OptunaHyperparameterTuner as Optimizer
from hyperparameter_tuning.sko_tuner import SKOHyperparameterTuner as Optimizer

def tune_hyperparameters( train_X, train_y, valid_X, valid_y,
        default_hps, hpt_specs, hpt_results_dir_path, best_hp_file_path, is_minimize=True):
    """ Tune hyperparameters
    """
    hyperparameter_tuner = Optimizer(default_hps, hpt_specs, hpt_results_dir_path, best_hp_file_path, is_minimize)
    best_hyperparams = hyperparameter_tuner.run_hyperparameter_tuning(train_X, train_y, valid_X, valid_y)
    return best_hyperparams