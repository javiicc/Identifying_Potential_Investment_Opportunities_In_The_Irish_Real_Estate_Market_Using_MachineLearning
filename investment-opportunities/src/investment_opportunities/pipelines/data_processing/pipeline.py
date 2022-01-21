"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.17.5
"""

from kedro.pipeline import Pipeline, node
from .nodes import preprocess_ads, get_db


def create_pipeline():
    return Pipeline(
        [
            node(
                func=get_db,
                inputs=None,
                outputs="advertisements",
                name="get_db_node",
            ),
            node(
                func=preprocess_ads,
                inputs="advertisements",
                outputs="preprocessed_ads",
                name="preprocess_ads_node",
            ),
        ]
    )
