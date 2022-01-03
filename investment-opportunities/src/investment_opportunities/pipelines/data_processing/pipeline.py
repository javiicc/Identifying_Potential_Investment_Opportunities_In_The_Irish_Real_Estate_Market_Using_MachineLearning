"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.17.5
"""

from kedro.pipeline import Pipeline, node
from .nodes import preprocess_ads


def create_pipeline():
    return Pipeline(
        [
            node(
                func=preprocess_ads,
                inputs="advertisements",
                outputs="preprocessed_ads",
                name="preprocess_ads_node",
            ),
        ]
    )
