from kedro.pipeline import Pipeline, node

from .nodes import temporal_process



def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=temporal_process,
                inputs="ads_with_location_features",
                outputs="temporal_input_data",
                name="temporal_input_model"
            ),
        ]
    )
