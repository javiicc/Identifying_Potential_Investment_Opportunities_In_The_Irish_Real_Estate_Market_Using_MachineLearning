
from kedro.pipeline import Pipeline, node

from .nodes import (evaluate_models) #voting_regresor
#get_voting_regressor_BA_estimator,


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=evaluate_models,
                inputs=["X_test", "y_test", "polyr", "knnr", "dtr", "voting_regressor_BA", "xgbr", "stackingr"],
                outputs=None,
                name="evaluate_models_node",
            ),
        ]
    )