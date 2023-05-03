## Introduction

This repository contains XAI code for a binary classification model. Both LIME and Shapley methods are implemented. The user can choose either of the two methods to explain the model predictions. Explanations are built on a random forest binary classifier model.

This repository is part of a tutorial series on Ready Tensor, a web platform for AI developers and users.

## Repository Contents

The `app/` folder in the repository contains the following key folders/sub-folders:

- `algorithm`: Contains the classifier implementation.
- `config`: Contains JSON configuration files for the model, hyperparameter tuning, and paths.
- `data_management/` will all files related to handling and preprocessing data.
- `inputs/` contains the input files related to the _titanic_ dataset.
- `model/` is a folder to save model artifacts and other assets specific to the trained model. Within this folder:
  - `artifacts/` is location to save model artifacts (i.e. the saved model including the trained preprocessing pipeline)
- `outputs/` is used to contain the predictions, logs and hpt results files. When the `predict.py` script is run, a predictions file called `predictions.csv` is saved in `outputs/predictions/` sub-directory.
- `app`: Main application directory.
  - `algorithm`: Contains the classifier implementation which uses random forest classifier built using the scikit-learn library.
  - `config`: Contains JSON configuration files for the model, hyperparameter tuning, and paths.
  - `data_management`: Contains code for data preprocessing, pipeline creation, custom transformers, and label encoding.
  - `hyperparameter_tuning`: Contains code for different hyperparameter tuning tools, including Scikit-Optimize, Hyperopt, and Optuna.
  - `inputs`: Contains the input data and data configuration files.
    - `data`: Contains the dataset for training and evaluation.
    - `data_config`: Contains JSON schema files for the dataset.
  - `model`: Contains the trained model artifacts.
  - `outputs`: Contains the output files, such as predictions and hyperparameter tuning results.
- `README.md`: This file.
- `requirements.txt`: Lists the required packages to run the code in this repository.
- `.gitignore`: Specifies files and folders to be ignored by Git.
- `LICENSE`: The license file for this repository.

See the following repository for information on the use of the data schema that is provided in the path `./app/inputs/data_config/`.

- https://github.com/readytensor/rt-tutorials-data-schema

See the following repository for more information on the data proprocessing logic defined in the path `./app/data_management/`.

- https://github.com/readytensor/rt-tutorials-data-preprocessing

See the following repository for more information on the random forest implementation.

- https://github.com/readytensor/rt-tutorials-oop-ml

See the following repository for more information on the HPT implementation within this repository:

- https://github.com/readytensor/rt-tutorials-hpt

## Usage

- Create your virtual environment and install dependencies listed in `requirements.txt`.
- Place the following 3 input files in the sub-directories in `./app/inputs/`:
  - Train data, which must be a CSV file, to be placed in `./app/inputs/data/training/`. File name can be any; extension must be ".csv".
  - Test data, which must be a CSV file, to be placed in `./app/inputs/data/testing/`. File name can be any; extension must be ".csv".
  - The schema file in JSON format , to be placed in `./app/inputs/data_config/`. The schema conforms to Ready Tensor specification for the **Binary Classification-Base** category. File name can be any; extension must be ".json".
- Run the `train.py` script to train the model, with optional flags for hyperparameter tuning or using default hyperparameters.
- Run the script `predict.py` to run test predictions using the trained model. This script will load the artifacts and create and save the predictions in a file called `predictions.csv` in the path `./app/outputs/predictions/`.

## Requirements

The code requires Python 3 and the following libraries:

```makefile
pandas==1.5.2
numpy==1.20.3
scikit-learn==1.0
feature-engine==1.2.0
imbalanced-learn==0.8.1
scikit-optimize==0.9.0
optuna==3.1.1
hyperopt==0.2.7
```

These packages can be installed by running the following command:

```python
pip install -r requirements.txt
```

Note that you only need to install the packages that are required for the hyperparameter tuning tool that you want to use. For example, if you want to use Scikit-Optimize, you only need to install the `scikit-optimize` package.
