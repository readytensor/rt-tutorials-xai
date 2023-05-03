import os

# Path to the root directory, which is `app` directory
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Path to config directory
CONFIG_DIR = os.path.join(ROOT_DIR, "config")
# Path to model config
MODEL_CONFIG_FILE_PATH  = os.path.join(CONFIG_DIR, "model_config.json")
# Path to hyperparameters file with default values
DEFAULT_HYPERPARAMETERS_FILE_PATH  = os.path.join(CONFIG_DIR, "default_hyperparameters.json")
# Path to hyperparameter tuning config file
HPT_CONFIG_FILE_PATH  = os.path.join(CONFIG_DIR, "hpt_config.json")

# Path to inputs
INPUT_DIR = os.path.join(ROOT_DIR, "inputs")
# File path for input schema file
SCHEMA_DIR = os.path.join(INPUT_DIR, "data_config")
# Path to data directory inside inputs directory
DATA_DIR = os.path.join(INPUT_DIR, "data")
# Path to training directory inside data directory
TRAIN_DIR = os.path.join(DATA_DIR, "training")
# Path to test directory inside data directory
TEST_DIR = os.path.join(DATA_DIR, "testing")

# Path to model directory
MODEL_PATH = os.path.join(ROOT_DIR, "model")
# Path to artifacts directory inside model directory
MODEL_ARTIFACTS_PATH = os.path.join(MODEL_PATH, "artifacts")
# Name of the preprocessing pipeline file
PIPELINE_FILE_PATH = os.path.join(MODEL_ARTIFACTS_PATH, "pipeline.joblib")
# Name of the label encoder file
LABEL_ENCODER_FILE_PATH = os.path.join(MODEL_ARTIFACTS_PATH, "label_encoder.joblib")
# Name of the classifier model file inside artifacts directory
CLASSIFIER_FILE_PATH = os.path.join(MODEL_ARTIFACTS_PATH, "classifier.joblib")
# Name of the best (tuned) hyperparameters file
BEST_HYPERPARAMETERS_FILE_PATH = os.path.join(MODEL_ARTIFACTS_PATH, "best_hyperparameters.json")

# Path to outputs
OUTPUT_DIR = os.path.join(ROOT_DIR, "outputs")
# Path to logs directory inside outputs directory
LOGS_DIR = os.path.join(OUTPUT_DIR, "logs")
# Path to predictions directory inside outputs directory
PREDICTIONS_DIR = os.path.join(OUTPUT_DIR, "predictions")
# Name of the file containing the predictions
PREDICTIONS_FILE_PATH = os.path.join(PREDICTIONS_DIR, "predictions.csv")
# Path to HPT results directory inside outputs directory
HPT_OUTPUTS_DIR = os.path.join(OUTPUT_DIR, "hpt_outputs")

# Log file paths
TRAIN_LOG_FILE_PATH = os.path.join(LOGS_DIR, "train_log.txt")
PREDICT_LOG_FILE_PATH = os.path.join(LOGS_DIR, "predict_log.txt")
SERVE_LOG_FILE_PATH = os.path.join(LOGS_DIR, "serve_log.txt")

# Error file paths
TRAIN_ERROR_FILE_PATH = os.path.join(LOGS_DIR, "train_error.txt")
PREDICT_ERROR_FILE_PATH = os.path.join(LOGS_DIR, "predict_error.txt")
SERVE_ERROR_FILE_PATH = os.path.join(LOGS_DIR, "serve_error.txt")
