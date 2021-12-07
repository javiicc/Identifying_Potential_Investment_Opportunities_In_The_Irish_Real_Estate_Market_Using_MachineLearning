from kedro.pipeline import Pipeline, node

from .nodes import location_feature_engineering



def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=location_feature_engineering,
                inputs="preprocessed_ads",
                outputs="ads_with_location_features",
                name="location_feature_engineering_node",
            ),

        ]
    )