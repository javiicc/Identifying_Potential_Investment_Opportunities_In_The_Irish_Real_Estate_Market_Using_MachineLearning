"""
This is a boilerplate pipeline 'algorithms'
generated using Kedro 0.17.5
"""

from kedro.pipeline import Pipeline, node
from .nodes import (get_levels, get_features_by_type, train_model, get_estimator)


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
                func=get_estimator,
                inputs=["levels_list", "num_features", "cat_features"],
                outputs="estimators_dict",
                name="pipe_estimator_node",
            ),
            node(
                func=train_model,
                inputs=["X_train", "y_train", "estimators_dict"],
                outputs=["polyr", "knnr", "dtr", "xgbr", "rfr"],
                name="training_node",
            ),
        ]
    )

