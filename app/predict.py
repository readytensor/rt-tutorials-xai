from utils import (
    load_data_schema,
    read_csv_in_directory, 
    save_dataframe_as_csv
)
from model_server import ModelServer
from config import paths


def run_batch_predictions():
    
    # loads the json file schema into a dictionary and use it to instantiate the schema provider
    data_schema = load_data_schema(paths.SCHEMA_DIR)

    # load test data
    test_data = read_csv_in_directory(file_dir_path=paths.TEST_DIR)

    # load model server
    model_server = ModelServer(
        pipeline_path=paths.PIPELINE_FILE_PATH,
        label_encoder_path=paths.LABEL_ENCODER_FILE_PATH,
        model_artifacts_path=paths.CLASSIFIER_FILE_PATH,
        data_schema=data_schema
    )

    # make predictions - these are predicted class probabilities
    predictions = model_server.predict_proba(data=test_data)

    # save_predictions
    save_dataframe_as_csv(
        dataframe=predictions,
        file_path=paths.PREDICTIONS_FILE_PATH,
    )
    
    print("Batch predictions completed successfully")


if __name__ == "__main__": 
    run_batch_predictions()