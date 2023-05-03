from typing import Dict
import joblib
from sklearn.pipeline import Pipeline
from feature_engine.wrappers import SklearnTransformerWrapper
from sklearn.preprocessing import StandardScaler
from feature_engine.encoding import RareLabelEncoder
from feature_engine.imputation import (
    AddMissingIndicator,
    CategoricalImputer,
    MeanMedianImputer    
)
from feature_engine.selection import (
    DropConstantFeatures,
    DropDuplicateFeatures,
    SmartCorrelatedSelection
)
from data_management import custom_transformers as transformers


def get_preprocess_pipeline(config: Dict, data_schema) -> Pipeline:
    """
    Create a preprocessor pipeline to transform data as defined by data_schema.
    """ 
    # Extract configuration values from the config dictionary
    preprocessing_config = config["preprocessing"]

    num_config = preprocessing_config["numeric_transformers"]
    clip_min_val = num_config["outlier_clipper"]["min_val"]
    clip_max_val = num_config["outlier_clipper"]["max_val"]
    imputation_method = num_config["mean_median_imputer"]["imputation_method"]
    
    cat_config = preprocessing_config["categorical_transformers"]
    cat_imputer_threshold = cat_config["cat_most_frequent_imputer"]["threshold"]
    rare_label_tol = cat_config["rare_label_encoder"]["tol"]
    rare_label_n_categories = cat_config["rare_label_encoder"]["n_categories"]
    
    feat_sel_pp_config = preprocessing_config["feature_selection_preprocessing"]
    constant_feature_tol = feat_sel_pp_config["constant_feature_dropper"]["tol"]
    correl_feature_threshold = feat_sel_pp_config["correlated_feature_dropper"]["threshold"]


    column_selector = transformers.ColumnSelector(columns=data_schema.features)
    string_caster = transformers.TypeCaster(
        vars=data_schema.categorical_features + [data_schema.id_field, data_schema.target_field],
        cast_type=str
    )
    float_caster = transformers.TypeCaster(
        vars=data_schema.numeric_features,
        cast_type=float
    )
    missing_indicator_numeric = AddMissingIndicator(variables=data_schema.numeric_features)
    mean_imputer_numeric = MeanMedianImputer(imputation_method=imputation_method, variables=data_schema.numeric_features)
    standard_scaler = SklearnTransformerWrapper(StandardScaler(), variables=data_schema.numeric_features)
    outlier_value_clipper = transformers.ValueClipper(
        fields_to_clip=data_schema.numeric_features,
        min_val=clip_min_val,
        max_val=clip_max_val
    )
    cat_most_frequent_imputer = transformers.MostFrequentImputer(
        cat_vars=data_schema.categorical_features,
        threshold=cat_imputer_threshold
    )
    cat_imputer_with_missing_tag = transformers.FeatureEngineCategoricalTransformerWrapper(
        transformer=CategoricalImputer,
        cat_vars=data_schema.categorical_features,
        imputation_method="missing"
    )
    rare_label_encoder = transformers.FeatureEngineCategoricalTransformerWrapper(
        transformer=RareLabelEncoder,
        cat_vars=data_schema.categorical_features,
        tol=rare_label_tol,
        n_categories=rare_label_n_categories
    )
    constant_feature_dropper = DropConstantFeatures(
        variables=None,
        tol=constant_feature_tol,
        missing_values="raise")
    duplicated_feature_dropper = DropDuplicateFeatures(
        variables=None,
        missing_values="raise")
    correlated_feature_dropper = SmartCorrelatedSelection(
        variables=None,
        selection_method="variance",
        threshold=correl_feature_threshold, missing_values="raise")
    one_hot_encoder = transformers.OneHotEncoderMultipleCols(ohe_columns=data_schema.categorical_features)
    
    
    pipeline = Pipeline([
        ("column_selector", column_selector),
        ("string_caster", string_caster),
        ("float_caster", float_caster),
        ("missing_indicator_numeric", missing_indicator_numeric),
        ("mean_imputer_numeric", mean_imputer_numeric),
        ("standard_scaler", standard_scaler),
        ("outlier_value_clipper", outlier_value_clipper),
        ("cat_most_frequent_imputer", cat_most_frequent_imputer),
        ("cat_imputer_with_missing_tag", cat_imputer_with_missing_tag),
        ("rare_label_encoder", rare_label_encoder),
        ("constant_feature_dropper", constant_feature_dropper),
        ("duplicated_feature_dropper", duplicated_feature_dropper),
        ("correlated_feature_dropper", correlated_feature_dropper),
        ("one_hot_encoder", one_hot_encoder)
    ])
    
    return pipeline


def save_pipeline(pipeline: Pipeline, file_path_and_name: str) -> None:
    """Save the fitted pipeline to a pickle file.

    Args:
        pipeline (Pipeline): The fitted pipeline to be saved.
        file_path_and_name (str): The path where the pipeline should be saved.
    """
    joblib.dump(pipeline, file_path_and_name)


def load_pipeline(file_path_and_name: str) -> Pipeline:
    """Load the fitted pipeline from the given path.

    Args:
        file_path_and_name: Path to the saved pipeline.

    Returns:
        Fitted pipeline.
    """
    return joblib.load(file_path_and_name)