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
