
"""Run the supervised model comparison pipeline."""

import mlflow

from src import config
from src.mlflow_tracking import configure_tracking
from src.supervised.data import N_FEATURES, load_scaled, select_features
from src.supervised.evaluation import build_results, evaluate, plot_confusion
from src.supervised.reporting import print_summary, save_outputs
from src.supervised.tracking import log_comparison, log_model_runs as _log_model_runs

from src.supervised.training import (
    K_CHOICES,
    TUNING_ROWS,
    WEIGHT_CHOICES,
    choose_k,
    train_models,
)
from src.supervised.xgboost_model import StringLabelXGBClassifier

def _run_models() -> None:

    # Load scaled training/test data, labels, and feature names
    X_train, y_train, X_test, y_test, cols = load_scaled()
    print(f"Train: {X_train.shape[0]:,} rows   Test: {X_test.shape[0]:,} rows")

    # Select the 20 most useful features using training data only
    selector, chosen = select_features(X_train, y_train, cols)

    X_train_all_features = X_train

    # Apply the same feature selection to training and test data
    X_train_selected = selector.transform(X_train)
    X_test_selected = selector.transform(X_test)
    print(f"Kept {len(chosen)} of {len(cols)} features")

    # Tune KNN hyperparameters using a validation split of training data
    print("\nChoosing k on a validation split of the training data ...")
    best_k, best_weights, table = choose_k(X_train_selected, y_train)
    print(table.sort_values("macro_f1", ascending=False).round(4).to_string(index=False))
    print(f"\nBest: k={best_k}, weights={best_weights}")

    # Train all classifiers and generate predictions on the test data
    estimators, predictions = train_models(
        X_train_selected,
        y_train,
        X_test_selected,
        best_k,
        best_weights,
    )

    # Calculate metrics and confusion matrices
    results = build_results(
        y_test, predictions, estimators, chosen,
        best_k, best_weights, table,
    )

    # Save results and print a readable performance summary
    save_outputs(results, estimators, selector, cols, table, y_test, predictions)
    print_summary(results)

    # Record parameters, metrics, models, and files in MLflow
    log_comparison(
        results=results,
        estimators=estimators,
        selector=selector,
        X_train=X_train_all_features,
        y_train=y_train,
        y_test=y_test,
        cols=cols,
        chosen=chosen,
        best_k=best_k,
        best_weights=best_weights,
        table=table,
    )


def main() -> None:
    # Configure MLflow, then create a parent run for this experiment
    configure_tracking()
    with mlflow.start_run(run_name="supervised-model-comparison"):
        _run_models()


# Run the pipeline when this file is executed directly
if __name__ == "__main__":
    main()
