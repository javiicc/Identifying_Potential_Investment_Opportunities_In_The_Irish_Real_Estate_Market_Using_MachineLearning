"""
This is a boilerplate pipeline 'voting_regressor_2_trained_with_all_data'
generated using Kedro 0.17.5
"""

from kedro.pipeline import Pipeline, node
from .nodes import voting_regresor_2_final_model


def create_pipeline():
    return Pipeline(
        [
            node(
                func=voting_regresor_2_final_model,
                inputs=["X", "y", "voting_regressor_BA", "rfr", "xgbr"],
                outputs="final_model",
                name="final_model_node",
            ),
        ]
    )
