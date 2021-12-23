from kedro.pipeline import Pipeline, node

from .nodes import (get_voting_regressor_BA_estimator)



def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=get_voting_regressor_BA_estimator,
                inputs=["polyr", "knnr", "dtr", "X_train", "y_train", "X_test", "y_test"],
                outputs="voting_regressor_BA",
                name="voting_regressor_BA_node",
            ),
        ]
    )