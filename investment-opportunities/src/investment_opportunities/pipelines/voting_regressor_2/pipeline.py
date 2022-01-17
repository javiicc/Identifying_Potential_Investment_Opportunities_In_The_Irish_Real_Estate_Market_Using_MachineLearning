"""
This is a boilerplate pipeline 'voting_regressor_2'
generated using Kedro 0.17.5
"""

from kedro.pipeline import Pipeline, node
from .nodes import (voting_regresor_2)


def create_pipeline():
    return Pipeline(
        [
            node(
                func=voting_regresor_2,
                inputs=["X_train", "y_train", "voting_regressor_BA", "rfr", "xgbr"],
                outputs="voting_regressor_2",
                name="voting_regressor_2_node",
            ),
        ]
    )
