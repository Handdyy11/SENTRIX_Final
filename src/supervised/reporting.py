
"""5. Save model outputs and print a readable evaluation summary."""

import json
import joblib
import pandas as pd

from src import config
from src.supervised.evaluation import plot_confusion


# Map each model to its output filename
MODEL_ARTIFACTS = {
    "knn": "knn_model.joblib",
    "decision_tree": "tree_model.joblib",
    "random_forest": "forest_model.joblib",
    "xgboost": "xgboost_model.joblib",
}


def save_outputs(results, estimators, selector, cols, table, y_test, predictions):
    """Save metrics, plots, tuning results, and trained models."""

    # Create the results folder if it doesn't exist
    config.RESULTS.mkdir(parents=True, exist_ok=True)

    metrics_path = config.RESULTS / "metrics_knn.json"
    confusion_path = config.RESULTS / "figures" / "confusion_matrix_knn.png"

    # Save evaluation metrics as JSON and the KNN confusion matrix as an image
    metrics_path.write_text(json.dumps(results, indent=2))
    plot_confusion(y_test, predictions["knn"], confusion_path)

    # Save each trained model with its feature selector and input feature names
    for name, filename in MODEL_ARTIFACTS.items():
        joblib.dump(
            {"model": estimators[name], "selector": selector, "features": cols},
            config.DATA_PROCESSED / filename,
        )

    # Save all KNN hyperparameter tuning results as a CSV file
    table.to_csv(config.RESULTS / "knn_tuning.csv", index=False)


def print_summary(results):
    """Print model scores, per-class metrics, and important features."""

    print("\nTest-set results")

    # Use readable names when displaying model results
    model_names = {
        "knn": "k-NN",
        "decision_tree": "decision tree",
        "random_forest": "random forest",
        "xgboost": "XGBoost",
        "majority_baseline": "always say Benign",
    }

    model_order = tuple(model_names)

    # Create a comparison table of accuracy and macro F1
    summary = pd.DataFrame(
        {
            "accuracy": [results[name]["accuracy"] for name in model_order],
            "macro_f1": [results[name]["macro_f1"] for name in model_order],
        },
        index=[model_names[name] for name in model_order],
    )
    print(summary.to_string())

    # Display precision and recall for each class for the main classifiers
    for model_name in ("knn", "decision_tree", "random_forest", "xgboost"):
        print(
            f"\n{model_names[model_name]}: recall and precision per family "
            "(recall = share of real attacks it caught)"
        )
        for family, values in results[model_name]["per_family"].items():
            print(
                f"  {family:<11} recall {values['recall']:.3f}   "
                f"precision {values['precision']:.3f}"
            )

    # Display the five most important features for tree-based models
    for model_name in ("random_forest", "xgboost"):
        print(f"\nTop 5 features by {model_names[model_name]} importance")
        for feature, weight in list(
            results[model_name]["feature_importance"].items()
        )[:5]:
            print(f"  {feature:<28}{weight:.4f}")
