from kedro.pipeline import Pipeline, node

from .nodes import (get_levels, get_features_by_type,
                    transformer_estimator, train_model, evaluate_model)



def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=get_levels,
                inputs="model_input_variables",
                outputs="levels_list",
                name="get_levels_node",
            ),
            node(
                func=get_features_by_type,
                inputs="model_input_variables",
                outputs=["num_features", "cat_features"],
                name="get_features_by_type_node",
            ),
            node(
                func=transformer_estimator,
                inputs=["levels_list", "num_features", "cat_features"],  #"regressor",
                outputs="pipe_estimator",
                name="pipe_estimator_node",
            ),
            node(
                func=train_model,
                inputs=["X_train", "y_train", "pipe_estimator"],
                outputs="regressor", # REGRESSOR
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