from kedro.pipeline import Pipeline, node

from .nodes import location_feature_engineering



def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=location_feature_engineering,
                inputs="df_without_outliers",
                outputs="ads_with_location_features",
                name="location_feature_engineering_node",
            ),

        ]
    )