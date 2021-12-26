from kedro.pipeline import Pipeline, node

from .nodes import merge_tables, split_data, variables_to_modelize



def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=merge_tables,
                inputs=["ads_with_location_features_and_geonames"], # Add DataFrames
                outputs="model_input",
                name="merge_node",
            ),
            node(
                func=variables_to_modelize,
                inputs="model_input",
                outputs="model_input_variables",
                name="select_variables_to_modelize_node",
            ),
            node(
                func=split_data,
                inputs="model_input_variables",
                outputs=["X_train", "X_test", "y_train", "y_test", "X", "y"],
                name="split_data_node",
            ),
        ]
    )