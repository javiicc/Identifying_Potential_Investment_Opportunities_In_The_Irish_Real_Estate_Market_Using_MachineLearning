from kedro.pipeline import Pipeline, node

from .nodes import (get_levels, get_features_by_type,
                    get_estimators)



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
                func=get_estimators,
                inputs=["levels_list", "num_features", "cat_features"],  #"regressor",
                outputs="estimators_pipe_list",
                name="estimator_pipe_node",
            ),
        ]
    )