"""
This is a boilerplate pipeline 'evaluate_models'
generated using Kedro 0.17.5
"""

from kedro.pipeline import Pipeline, node
from .nodes import (evaluate_models)


def create_pipeline():
    return Pipeline(
        [
            node(
                func=evaluate_models,
                inputs=["X_test", "y_test", "polyr", "knnr", "dtr", "voting_regressor_BA",
                        "xgbr", "rfr", "voting_regressor_2"],
                outputs=None,
                name="evaluate_models_node",
            ),
        ]
    )

