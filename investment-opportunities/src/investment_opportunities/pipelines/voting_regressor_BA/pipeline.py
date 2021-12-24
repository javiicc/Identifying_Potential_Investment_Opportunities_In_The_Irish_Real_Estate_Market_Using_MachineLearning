
from kedro.pipeline import Pipeline, node

from .nodes import (voting_regresor)
#get_voting_regressor_BA_estimator,


def create_pipeline(**kwargs):
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