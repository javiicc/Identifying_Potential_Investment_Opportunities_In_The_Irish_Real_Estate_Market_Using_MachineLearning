"""
This is a boilerplate pipeline 'voting_regressor_BA'
generated using Kedro 0.17.5
"""

from kedro.pipeline import Pipeline, node
from .nodes import (voting_regresor)


def create_pipeline():
    return Pipeline(
        [
            node(
                func=voting_regresor,
                inputs=["X_train", "y_train", "polyr", "knnr", "dtr"],
                outputs="voting_regressor_BA",
                name="voting_regressor_node",
            ),
        ]
    )
