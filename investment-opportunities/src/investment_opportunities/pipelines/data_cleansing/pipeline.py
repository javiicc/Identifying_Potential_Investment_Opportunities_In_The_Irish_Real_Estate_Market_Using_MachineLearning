"""
This is a boilerplate pipeline 'data_cleansing'
generated using Kedro 0.17.5
"""

from kedro.pipeline import Pipeline, node
from .nodes import drop_outliers


def create_pipeline():
    return Pipeline(
        [
            node(
                func=drop_outliers,
                inputs="preprocessed_ads",
                outputs="df_no_outliers",
                name="drop_outliers_node",
            ),
        ]
    )
