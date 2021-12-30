
from kedro.pipeline import Pipeline, node

from .nodes import (get_stacking_final_model)
#get_voting_regressor_BA_estimator,


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=get_stacking_final_model,
                inputs=["voting_regressor_BA", "xgbr", "X", "y"],
                outputs="final_model",
                name="final_model_node",
            ),
        ]
    )