"""
This is a boilerplate pipeline 'stacking_regressor'
generated using Kedro 0.17.5
"""

from kedro.pipeline import Pipeline, node
from .nodes import (get_stacking)


def create_pipeline():
    return Pipeline(
        [
            node(
                func=get_stacking,
                inputs=["voting_regressor_BA", "xgbr", "X_train", "y_train"],
                outputs="stackingr",
                name="stacking_regressor_node",
            ),
        ]
    )
