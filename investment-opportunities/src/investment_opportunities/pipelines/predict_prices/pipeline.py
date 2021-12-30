from kedro.pipeline import Pipeline, node

from .nodes import (get_predictions, get_residuals, add_features_for_frontend)


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=get_predictions,
                inputs=["final_model", "X", "y"],
                outputs="data_w_predictions",
                name="get_predictions_node",
            ),
            node(
                func=get_residuals,
                inputs=["data_w_predictions"],
                outputs="data_w_residuals",
                name="get_residuals_node",
            ),
            node(
                func=add_features_for_frontend,
                inputs=["data_w_residuals", "model_input"],
                outputs="data_for_frontend",
                name="add_features_for_frontend_node",
            ),
        ]
    )