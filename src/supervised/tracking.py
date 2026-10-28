
"""6. Log comparison metrics, model artifacts, and child runs to MLflow."""

import mlflow
from sklearn.pipeline import Pipeline

from src import config
from src.supervised.training import K_CHOICES, TUNING_ROWS, WEIGHT_CHOICES


def log_model_runs(models, results, selector, X_train, cols) -> None:
    """Log each trained model and its test metrics as a child run."""

    for model_name, estimator in models.items():
        model_results = results[model_name]

        # Combine feature selection and classification into one pipeline
        pipeline = Pipeline(
            steps=[
                ("feature_selector", selector),
                ("classifier", estimator)
            ]
        )

        # Create a separate child run for this model
        with mlflow.start_run(run_name=model_name, nested=True):
            mlflow.set_tags({
                "model_name": model_name,
                "model_library": "scikit-learn",
            })

            # Record training details and model hyperparameters
            mlflow.log_params({
                "training_rows": len(X_train),
                "input_feature_count": len(cols),
                "selected_feature_count": selector.get_support().sum(),
                **{
                    f"estimator_{name}": str(value)
                    for name, value in estimator.get_params(deep=False).items()
                },
            })

            # Record test performance metrics
            mlflow.log_metrics({
                "test_accuracy": model_results["accuracy"],
                "test_macro_f1": model_results["macro_f1"],
                **{
                    f"test_{family}_{metric}": value
                    for family, family_metrics in model_results["per_family"].items()
                    for metric, value in family_metrics.items()
                },
            })

            # Save the pipeline to MLflow so it can be retrieved later
            mlflow.sklearn.log_model(
                sk_model=pipeline,
                name="model",
                serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
                input_example=X_train[:5],
            )


def log_comparison(
    results, estimators, selector, X_train, y_train, y_test,
    cols, chosen, best_k, best_weights, table,
):
    """Log the overall experiment and its results."""

    # Record dataset details, selected features, and model settings
    mlflow.log_params({
        "random_seed": config.SEED,
        "training_rows": len(y_train),
        "testing_rows": len(y_test),
        "input_feature_count": len(cols),
        "selected_feature_count": len(chosen),
        "selected_features": ", ".join(chosen),
        "knn_k": best_k,
        "knn_weights": best_weights,
        "knn_candidate_k": ", ".join(map(str, K_CHOICES)),
        "knn_candidate_weights": ", ".join(WEIGHT_CHOICES),
        "tuning_rows_limit": TUNING_ROWS,
        "decision_tree_max_depth": 12,
        "random_forest_n_estimators": 200,
        "random_forest_min_samples_leaf": 2,
        "random_forest_class_weight": "balanced_subsample",
        "xgboost_n_estimators": 300,
        "xgboost_max_depth": 8,
        "xgboost_learning_rate": 0.1,
        "xgboost_subsample": 0.9,
        "xgboost_colsample_bytree": 0.9,
        "xgboost_sample_weight": "balanced",
    })

    # Log accuracy, macro F1, precision, recall, and F1 for each model
    for model_name in (
        "knn", "decision_tree", "random_forest",
        "xgboost", "majority_baseline",
    ):
        model_results = results[model_name]

        mlflow.log_metrics({
            f"{model_name}_accuracy": model_results["accuracy"],
            f"{model_name}_macro_f1": model_results["macro_f1"],
        })

        for family, metrics in model_results["per_family"].items():
            mlflow.log_metrics({
                f"{model_name}_{family}_precision": metrics["precision"],
                f"{model_name}_{family}_recall": metrics["recall"],
                f"{model_name}_{family}_f1": metrics["f1-score"],
            })

    # Log each trained model as a separate child run
    log_model_runs(estimators, results, selector, X_train, cols)

    # Upload saved results, plots, tuning data, and model files to MLflow
    artifacts = (
        config.RESULTS / "metrics_knn.json",
        config.RESULTS / "figures" / "confusion_matrix_knn.png",
        config.RESULTS / "knn_tuning.csv",
        config.DATA_PROCESSED / "knn_model.joblib",
        config.DATA_PROCESSED / "tree_model.joblib",
        config.DATA_PROCESSED / "forest_model.joblib",
        config.DATA_PROCESSED / "xgboost_model.joblib",
    )

    for artifact in artifacts:
        mlflow.log_artifact(
            str(artifact),
            artifact_path="model-results"
        )

    print(f"\nSaved {config.RESULTS / 'metrics_knn.json'}")
    print(f"Saved {config.RESULTS / 'figures' / 'confusion_matrix_knn.png'}")
