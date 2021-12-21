"""Project pipelines."""

from typing import Dict

from kedro.pipeline import Pipeline

from .pipelines import data_processing as dp
from .pipelines import data_cleansing as dc
from .pipelines import feature_engineering_geospatial_data as fe
from .pipelines import merge as m
from .pipelines import data_science as ds


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipeline.

    Returns:
    A mapping from a pipeline name to a ``Pipeline`` object.

    """
    data_processing_pipeline = dp.create_pipeline()
    data_cleansing_pipeline = dc.create_pipeline()
    feature_engineering_pipeline = fe.create_pipeline()
    merge_tables_pipeline = m.create_pipeline()
    data_science_pipeline = ds.create_pipeline()

    return {
        "__default__": data_processing_pipeline + data_cleansing_pipeline
                       + feature_engineering_pipeline + merge_tables_pipeline
                       + data_science_pipeline,
        "dp": data_processing_pipeline,
        "dc": data_cleansing_pipeline,
        "fe": feature_engineering_pipeline,
        "m": merge_tables_pipeline,
        "ds": data_science_pipeline,
    }
