## Introduction

This repository is part of a tutorial series on Ready Tensor, a web platform for AI developers and users. It is referenced in the tutorial called **Enhancing Model Interpretability with Shapley Values: An Explainable AI (XAI) Approach**. The purpose of the tutorial series is to help AI developers create adaptable algorithm implementations that avoid hard-coding your logic to a specific dataset. This makes it easier to re-use your algorithms with new datasets in the future without requiring any code change.

## Repository Contents

```bash
binary_class_project/
├── examples/
│   ├── titanic_schema.json
│   ├── titanic_train.csv
│   └── titanic_test.csv
├── inputs/
│   ├── data/
│   │   ├── testing/
│   │   └── training/
│   └── schema/
├── model/
│   └── artifacts/
├── outputs/
│   ├── errors/
│   ├── hpt_outputs/
│   └── predictions/
├── src/
│   ├── config/
│   │   ├── default_hyperparameters.json
│   │   ├── hpt.json
│   │   ├── model_config.json
│   │   ├── paths.py
│   │   └── preprocessing.json
│   ├── data_models/
│   ├── hyperparameter_tuning/
│   │   ├── __init__.json
│   │   └── tuner.py
│   ├── prediction/
│   │   ├── __init__.json
│   │   └── predictor_model.py
│   ├── preprocessing/
│   │   ├── custom_transformers.py
│   │   ├── pipeline.py
│   │   ├── preprocess.py
│   │   └── target_encoder.py
│   ├── schema/
│   │   └── data_schema.py
│   ├── xai/
│   │   ├── __init__.json
│   │   └── explainer.py
│   ├── predict.py
│   ├── serve.py
│   ├── train.py
│   └── utils.py
├── tests/
│   ├── integration_tests/
│   ├── performance_tests/
│   └── unit_tests/
│       ├── <mirrors /src structure>
│       └── ...
├── tmp/
├── .gitignore
├── LICENSE
├── pytest.ini
├── README.md
├── requirements.txt
└── requirements-test.txt
```

- **`/examples`**: This directory contains example files for the titanic dataset. Three files are included: `titanic_schema.json`, `titanic_train.csv` and `titanic_test.csv`. You can place these files in the `inputs/schema`, `inputs/data/training` and `inputs/data/testing` folders, respectively.
- **`/inputs`**: This directory contains all the input files for your project, including the data and schema files. The data is further divided into testing and training subsets.
- **`/model/artifacts`**: This directory is used to store the model artifacts, such as trained models and their parameters.
- **`/outputs`**: The outputs directory contains sub-directories for error logs, and hyperparameter tuning outputs, and prediction results. Note that model artifacts should not be saved in this directory. Instead, they should be saved in the `/model/artifacts` directory.
- **`/src`**: This directory holds the source code for the project. It is further divided into various subdirectories such as `config` for configuration files, `data_models` for data models for input validation, `hyperparameter_tuning` for hyperparameter-tuning (HPT) related files, `prediction` for prediction model scripts, `preprocessing` for data preprocessing scripts, `schema` for schema scripts, and `xai` for explainable AI scripts.
  - The script called `explainer.py` under `src/xai/` is used to implement the shap explainer.
  - In the script `serve.py` under `src`, the `explain` endpoint is defined which provides local explanations for the predictions. The service runs on port 8080.
- **`/tests`**: This directory contains all the tests for the project. It contains sub-directories for specific types of tests such as unit tests, integration tests, and performance tests. For unit tests, the directory structure mirrors the `/src` directory structure.
- **`/tmp`**: This directory is used for storing temporary files which are not necessary to commit to the repository.
- **`.gitignore`**: This file specifies the files and folders that should be ignored by Git.
- **`LICENSE`**: This file contains the license for the project.
- **`README.md`**: This file contains the documentation for the project, explaining how to set it up and use it.
- **`requirements.txt`**: This file lists the dependencies for the project, making it easier to install all necessary packages.

## Usage

- Create your virtual environment and install dependencies listed in `requirements.txt`.
- Move the three example files (`titanic_schema.json`, `titanic_train.csv` and `titanic_test.csv`) into the `inputs/schema`, `inputs/data/training` and `inputs/data/testing` folders, respectively.
- Run the `train.py` script to train the model, with `--tune` or `-t` flag for hyperparameter tuning. If the flag is not provided, the model will be trained with default hyperparameters. This will save the model artifacts, including the preprocessing pipeline and label encoder, and the explainer, in the path `./model/artifacts/`. When tuning is requested, the hyperparameter tuning results will be saved in a file called `hpt_results.csv` in the path `./outputs/hpt_outputs/`. The best hyperparameters are used in the trained model.
- Run the script `predict.py` to run test predictions using the trained model. This script will load the artifacts and create and save the predictions in a file called `predictions.csv` in the path `./outputs/predictions/`.
- Run the script `serve.py` to start the inference service, which can be queried using the `/ping`, `/infer` and `/explain` endpoints.

**Getting predictions ** <br/>
To get predictions for a single sample, use the following command:

```bash
curl -X POST -H "Content-Type: application/json" -d '{
  {
    "instances": [
        {
            "PassengerId": "879",
            "Pclass": 3,
            "Name": "Laleff, Mr. Kristo",
            "Sex": "male",
            "Age": None,
            "SibSp": 0,
            "Parch": 0,
            "Ticket": "349217",
            "Fare": 7.8958,
            "Cabin": None,
            "Embarked": "S"
        }
    ]
}' http://localhost:8080/infer
```

The key `instances` contains a list of objects, each of which is a sample for which the prediction is requested. The server will respond with a JSON object containing the predicted probabilities for each input record:

```json
{
  "status": "success",
  "message": "",
  "timestamp": "<timestamp>",
  "requestId": "<uniquely generated id>",
  "targetClasses": ["0", "1"],
  "targetDescription": "A binary variable indicating whether or not the passenger survived (0 = No, 1 = Yes).",
  "predictions": [
    {
      "sampleId": "879",
      "predictedClass": "0",
      "predictedProbabilities": [0.97548, 0.02452]
    }
  ]
}
```

**Getting predictions and local explanations ** <br/>
To get predictions and explanations for a single sample, use the following command:

```bash
curl -X POST -H "Content-Type: application/json" -d '{
  {
    "instances": [
        {
            "PassengerId": "879",
            "Pclass": 3,
            "Name": "Laleff, Mr. Kristo",
            "Sex": "male",
            "Age": None,
            "SibSp": 0,
            "Parch": 0,
            "Ticket": "349217",
            "Fare": 7.8958,
            "Cabin": None,
            "Embarked": "S"
        }
    ]
}' http://localhost:8080/explain
```

The server will respond with a JSON object containing the predicted probabilities and locations for each input record:

```json
{
  "status": "success",
  "message": "",
  "timestamp": "2023-05-22T10:51:45.860800",
  "requestId": "0ed3d0b76d",
  "targetClasses": ["0", "1"],
  "targetDescription": "A binary variable indicating whether or not the passenger survived (0 = No, 1 = Yes).",
  "predictions": [
    {
      "sampleId": "879",
      "predictedClass": "0",
      "predictedProbabilities": [0.92107, 0.07893],
      "explanation": {
        "baseline": [0.57775, 0.42225],
        "featureScores": {
          "Age_na": [0.05389, -0.05389],
          "Age": [0.02582, -0.02582],
          "SibSp": [-0.00469, 0.00469],
          "Parch": [0.00706, -0.00706],
          "Fare": [0.05561, -0.05561],
          "Embarked_S": [0.01582, -0.01582],
          "Embarked_C": [0.00393, -0.00393],
          "Embarked_Q": [0.00657, -0.00657],
          "Pclass_3": [0.0179, -0.0179],
          "Pclass_1": [0.02394, -0.02394],
          "Sex_male": [0.13747, -0.13747]
        }
      }
    }
  ],
  "explanationMethod": "Shap"
}
```

## OpenAPI

Since the service is implemented using FastAPI, we get automatic documentation of the APIs offered by the service. Visit the docs at `http://localhost:8080/docs`.

## Requirements

Dependencies are listed in the file `requirements.txt`. These packages can be installed by running the following command:

```python
pip install -r requirements.txt
```

For testing, dependencies are listed in the file `requirements-test.txt`. You can install these packages by running the following command:

```python
pip install -r requirements-test.txt
```
