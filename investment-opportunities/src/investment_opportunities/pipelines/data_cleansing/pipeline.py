
from kedro.pipeline import Pipeline, node

from .nodes import drop_outliers


def create_pipeline(**kwargs):
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