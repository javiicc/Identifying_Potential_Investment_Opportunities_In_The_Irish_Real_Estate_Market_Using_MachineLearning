from kedro.pipeline import Pipeline, node

from .nodes import merge_tables



def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=merge_tables,
                inputs=["ads_with_location_features_and_geonames"], # Add DataFrames
                outputs="model_input",
                name="merge_node",
            ),
        ]
    )