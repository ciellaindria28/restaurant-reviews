from kedro.pipeline import Pipeline, node, pipeline
from .nodes import make_predictions, preprocess_reviews,extract_features, split_data,report_accuracy


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=preprocess_reviews,
                inputs=["restaurant_reviews_dataset"],
                outputs="corpus",
                name="preprocess_reviews",
            ),
            node(
                func=extract_features,
                inputs=["restaurant_reviews_dataset","corpus"],
                outputs=["X","y"],
                name="extract_features",
            ),
            node(
                func=split_data,
                inputs=["X","y"],
                outputs=["X_train", "X_test", "y_train", "y_test"],
                name="split",
            ),
            node(
                func=make_predictions,
                inputs=["X_train", "X_test", "y_train"],
                outputs="y_pred",
                name="make_predictions",
            ),
            node(
                func=report_accuracy,
                inputs=["y_pred", "y_test"],
                outputs=None,
                name="report_accuracy",
            ),
        ]
    )
