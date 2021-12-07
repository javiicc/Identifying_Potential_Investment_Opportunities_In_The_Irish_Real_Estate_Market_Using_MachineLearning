from kedro.pipeline import Pipeline, node

from .nodes import preprocess_ads, drop_outliers, drop_outliers_ix



def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=preprocess_ads,
                inputs="advertisements",
                outputs="preprocessed_ads",
                name="preprocess_ads_node",
            ),
            node(
                func=drop_outliers,
                inputs="preprocessed_ads",
                outputs="outliers_dict",
                name="list_outliers_node",
            ),
            node(
                func=drop_outliers_ix,
                inputs=["preprocessed_ads", "outliers_dict"],
                outputs="df_without_outliers",
                name="drop_outliers_node",
            ),
        ]
    )