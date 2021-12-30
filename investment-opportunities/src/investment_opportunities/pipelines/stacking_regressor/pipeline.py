
from kedro.pipeline import Pipeline, node

from .nodes import (get_stacking)
#get_voting_regressor_BA_estimator,


def create_pipeline(**kwargs):
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