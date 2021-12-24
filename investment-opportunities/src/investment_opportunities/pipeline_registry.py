"""Project pipelines."""

from typing import Dict

from kedro.pipeline import Pipeline

from .pipelines import data_processing as dp
from .pipelines import data_cleansing as dc
from .pipelines import feature_engineering_geospatial_data as fe
from .pipelines import model_imput as mi
from .pipelines import basic_algorithms as ba
from .pipelines import voting_regressor_BA as vr
from .pipelines import evaluate_models as em

def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipeline.

    Returns:
    A mapping from a pipeline name to a ``Pipeline`` object.

    """
    data_processing_pipeline = dp.create_pipeline()
    data_cleansing_pipeline = dc.create_pipeline()
    feature_engineering_pipeline = fe.create_pipeline()
    model_input_pipeline = mi.create_pipeline()
    basic_algorithms_pipeline = ba.create_pipeline()
    voting_regressor_BA_pipeline = vr.create_pipeline()
    evaluate_models_pipeline = em.create_pipeline()

    return {
        "__default__": data_processing_pipeline + data_cleansing_pipeline
                       + feature_engineering_pipeline + model_input_pipeline
                       + basic_algorithms_pipeline + voting_regressor_BA_pipeline
                       + evaluate_models_pipeline,
        "dp": data_processing_pipeline,
        "dc": data_cleansing_pipeline,
        "fe": feature_engineering_pipeline,
        "mi": model_input_pipeline,
        "pr": basic_algorithms_pipeline,
        "vr": voting_regressor_BA_pipeline,
        "em": evaluate_models_pipeline,
    }
