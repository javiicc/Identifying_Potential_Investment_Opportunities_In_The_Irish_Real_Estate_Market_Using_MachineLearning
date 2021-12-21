from kedro.pipeline import Pipeline, node

from .nodes import (variables_to_modelize, split_data, get_levels, get_features_by_type,
                    transformer_estimator, train_model, evaluate_model)



def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=variables_to_modelize,
                inputs="model_input",
                outputs="model_input_variables",
                name="select_variables_to_modelize_node",
            ),
            node(
                func=get_levels,
                inputs="model_input_variables",
                outputs=["levels_type_house", "levels_code"],
                name="get_levels_node",
            ),
            node(
                func=get_features_by_type,
                inputs="model_input_variables",
                outputs=["num_features", "cat_features"],
                name="get_features_by_type_node",
            ),
            node(
                func=split_data,
                inputs="model_input_variables",
                outputs=["X_train", "X_test", "y_train", "y_test"],
                name="split_data_node",
            ),
            node(
                func=transformer_estimator,
                inputs=["levels_type_house", "levels_code", "num_features", "cat_features"],
                outputs="pipe_estimator",
                name="pipe_estimator_node",
            ),
            node(
                func=train_model,
                inputs=["X_train", "y_train", "pipe_estimator"],
                outputs="regressor",
                name="regressor_node",
            ),
            node(
                func=evaluate_model,
                inputs=["regressor", "X_test", "y_test"],
                outputs=None,
                name="evaluate_model_node",
            ),
        ]
    )