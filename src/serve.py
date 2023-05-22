from fastapi import FastAPI, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Tuple, Any
import pandas as pd
import uuid
import uvicorn

from schema.data_schema import load_saved_schema
from config import paths
from preprocessing.preprocess import (
    load_pipeline_and_target_encoder,
    transform_data
)
from prediction.predictor_model import load_predictor_model
from  predict import get_model_predictions, add_ids_to_predictions
from utils import read_json_as_dict
from xai.explainer import load_explainer, get_explanations_from_explainer


# Create an instance of the FastAPI class
app = FastAPI()

class ModelResources:
    def __init__(
            self,
            saved_schema_path: str = paths.SAVED_SCHEMA_PATH,
            predictor_file_path: str = paths.PREDICTOR_FILE_PATH,
            pipeline_file_path: str = paths.PIPELINE_FILE_PATH,
            target_encoder_file_path: str = paths.TARGET_ENCODER_FILE_PATH,
            model_config_file_path: str = paths.MODEL_CONFIG_FILE_PATH,
            explainer_file_path: str = paths.EXPLAINER_FILE_PATH,
        ):
        self.data_schema = load_saved_schema(saved_schema_path)
        self.predictor_model = load_predictor_model(predictor_file_path)
        self.preprocessor, self.target_encoder = load_pipeline_and_target_encoder(
            pipeline_file_path,
            target_encoder_file_path
        )
        self.model_config = read_json_as_dict(model_config_file_path)
        self.explainer = load_explainer(explainer_file_path)


def get_model_resources():
    """Returns an instance of ModelResources."""
    return ModelResources()


@app.get("/ping")
async def ping() -> dict:
    """GET endpoint that returns a message indicating the service is running.

    Returns:
        dict: A dictionary with a "message" key and "Pong!" value.
    """
    return {"message": "Pong!"}


def generate_unique_request_id():
    """Generates unique alphanumeric id"""
    return uuid.uuid4().hex[:10]


def create_sample_prediction(sample: dict, id_field: str, class_names: List[str]) -> dict:
    """
    Create a dictionary with the prediction results for a single sample.

    Args:
        sample (dict): A single sample's prediction results.
        id_field (str): The name of the field containing the sample ID.
        class_names (list): The names of the target classes.

    Returns:
        dict: A dictionary containing the prediction results for the sample.
    """
    return {
        "sampleId": sample[id_field],
        "predictedClass": str(sample["__predicted_class"]),
        "predictedProbabilities": [
            round(sample[class_names[0]], 5),
            round(sample[class_names[1]], 5)
        ]
    }


def create_predictions_response(
        predictions_df: pd.DataFrame,
        data_schema: Any
        ) -> None:
    """
    Convert the predictions DataFrame to a response dictionary in required format.

    Args:
        transformed_data (pd.DataFrame): The transfomed input data for prediction.
        data_schema (Any): An instance of the BinaryClassificationSchema.

    Returns:
        dict: The response data in a dictionary.
    """
    class_names = data_schema.target_classes
    # find predicted class which has the highest probability
    predictions_df["__predicted_class"] = predictions_df[class_names].idxmax(axis=1)
    sample_predictions = [
        create_sample_prediction(sample, data_schema.id, class_names)
        for sample in predictions_df.to_dict(orient="records")
    ]
    predictions_response = {
        "status": "success",
        "message": "",
        "timestamp": pd.Timestamp.now().isoformat(),
        "requestId": generate_unique_request_id(),
        "targetClasses": class_names,
        "targetDescription": data_schema.target_description,
        "predictions": sample_predictions,
    }    
    return predictions_response


class InferenceRequestBodyModel(BaseModel):
    """
    A Pydantic BaseModel for handling inference requests.

    Attributes:
        instances (list): A list of input data instances.
    """
    instances: List[dict]


async def transform_req_data_and_make_predictions(
    request: InferenceRequestBodyModel,
    model_resources: ModelResources
) -> Tuple[pd.DataFrame, dict]:
    """Transform request data and generate predictions based on request.

    Args:
        request (InferenceRequestBodyModel): The request body containing the input data.
        model_resources (ModelResources): Resources needed by inference service.

    Returns:
        Tuple[pd.DataFrame, dict]: Tuple containing transformed data and prediction response.
    """
    data = pd.DataFrame.from_records(request.dict()["instances"])
    print(f"Predictions requested for {len(data)} samples.")
    transformed_data, _ = transform_data(
        model_resources.preprocessor, model_resources.target_encoder, data)
    predictions_df = get_model_predictions(
        transformed_data,
        model_resources.predictor_model,
        model_resources.data_schema.target_classes,
        model_resources.model_config["prediction_field_name"],
        return_probs=True)
    predictions_df_with_ids = add_ids_to_predictions(
        data, predictions_df, model_resources.data_schema.id)
    predictions_response = create_predictions_response(
        predictions_df_with_ids,
        model_resources.data_schema
    )
    return transformed_data, predictions_response


@app.post("/infer", tags=["inference"], response_class=JSONResponse)
async def infer(request: InferenceRequestBodyModel,
                model_resources: ModelResources = Depends(get_model_resources)) -> dict:
    """POST endpoint that takes input data as a JSON object and returns
       predicted class probabilities.

    Args:
        request (InferenceRequestBodyModel): The request body containing the input data.
        model (ModelResources, optional): The model resources instance. Defaults to Depends(get_model).

    Raises:
        HTTPException: If there is an error during inference.

    Returns:
        dict: A dictionary with "status", "message", and "predictions" keys.
    """
    _, predictions_response = await transform_req_data_and_make_predictions(request, model_resources)
    return predictions_response


def update_predictions_response_with_explanations(
        predictions_response: dict, explanations: dict) -> dict:
    """
    Update the predictions response with explanations.

    Inserts explanations for each sample into the respective prediction dictionary
    for the sample.

    Args:
        predictions_response (dict): The response data in a dictionary.
        explanations (dict): The explanations for the predictions.
    """
    for pred, exp in zip(predictions_response["predictions"], explanations["explanations"]):
        pred["explanation"] = exp
    predictions_response["explanationMethod"] = explanations["explanation_method"]
    return predictions_response


@app.post("/explain", tags=["explanations", "XAI"], response_class=JSONResponse)
async def explain(request: InferenceRequestBodyModel,
                  model_resources: ModelResources = Depends(get_model_resources)) -> dict:
    """POST endpoint that takes input data as a JSON object and returns
       the predicted class probabilities with explanations.

    Args:
        request (InferenceRequestBodyModel): The request body containing the input data.
        model_resources (ModelResources, optional): The model resources instance.
                                        Defaults to Depends(get_model_resources).

    Raises:
        HTTPException: If there is an error during inference.

    Returns:
        dict: A dictionary with "status", "message", "timestamp", "requestId",
                "targetClasses", "targetDescription", "predictions", and "explanationMethod" keys.
    """
    transformed_data, predictions_response = \
        await transform_req_data_and_make_predictions(request, model_resources)
    explanations = get_explanations_from_explainer(
        instances_df=transformed_data,
        explainer=model_resources.explainer,
        predictor_model=model_resources.predictor_model,
        class_names=model_resources.data_schema.target_classes
    )
    predictions_response = update_predictions_response_with_explanations(
        predictions_response=predictions_response,
        explanations=explanations
    )
    return predictions_response


if __name__ == "__main__":
    print("Starting service. Listening on port 8080.")
    uvicorn.run(app, host="0.0.0.0", port=8080)
