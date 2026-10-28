
"""4. Calculate model metrics and create evaluation results."""

import matplotlib
matplotlib.use("Agg")  # Save plots without opening a window

import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from src import config


def evaluate(y_true, y_pred):
    """Calculate model performance metrics."""

    # Compare actual labels with predictions and get per-class metrics
    report = classification_report(
        y_true, y_pred, output_dict=True, zero_division=0
    )

    # Extract precision, recall, and F1-score for each class
    per_family = {
        name: {
            metric: round(values[metric], 4)
            for metric in ("precision", "recall", "f1-score")
        }
        for name, values in report.items()
        if name not in ("accuracy", "macro avg", "weighted avg")
    }

    # Return overall accuracy, macro F1, and per-class metrics
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "macro_f1": round(f1_score(y_true, y_pred, average="macro"), 4),
        "per_family": per_family,
    }


def build_results(y_test, predictions, estimators, chosen, best_k, best_weights, table):
    """Collect evaluation results for all classifiers."""

    # Keep class labels present in actual or predicted test data
    labels = [
        family for family in config.FAMILIES
        if family in set(y_test) | set(predictions["knn"])
    ]

    results = {
        "majority_baseline": evaluate(
            y_test, predictions["majority_baseline"]
        ),

        "decision_tree": {
            **evaluate(y_test, predictions["decision_tree"]),
            # Show correct and incorrect predictions for each class
            "confusion": confusion_matrix(
                y_test, predictions["decision_tree"], labels=labels
            ).tolist(),
        },

        "random_forest": {
            **evaluate(y_test, predictions["random_forest"]),
            "confusion": confusion_matrix(
                y_test, predictions["random_forest"], labels=labels
            ).tolist(),

            # Pair feature names with importance scores and sort descending
            "feature_importance": {
                feature: round(float(weight), 4)
                for feature, weight in sorted(
                    zip(chosen, estimators["random_forest"].feature_importances_),
                    key=lambda pair: -pair[1],
                )
            },
        },

        "xgboost": {
            **evaluate(y_test, predictions["xgboost"]),
            "confusion": confusion_matrix(
                y_test, predictions["xgboost"], labels=labels
            ).tolist(),

            "feature_importance": {
                feature: round(float(weight), 4)
                for feature, weight in sorted(
                    zip(chosen, estimators["xgboost"].feature_importances_),
                    key=lambda pair: -pair[1],
                )
            },
        },

        "knn": {
            "k": best_k,
            "weights": best_weights,
            **evaluate(y_test, predictions["knn"]),
            "confusion": confusion_matrix(
                y_test, predictions["knn"], labels=labels
            ).tolist(),
        },

        # Save class labels, tuning results, and features used
        "labels": labels,
        "tuning": table.round(4).to_dict("records"),
        "features_used": chosen,
    }

    return results


def plot_confusion(y_true, y_pred, path):
    """Create and save a normalized confusion matrix plot."""

    labels = [
        family for family in config.FAMILIES
        if family in set(y_true) | set(y_pred)
    ]

    counts = confusion_matrix(y_true, y_pred, labels=labels)

    # Convert counts to proportions for each actual class
    shares = counts / counts.sum(axis=1, keepdims=True).clip(min=1)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(shares, cmap="Blues", vmin=0, vmax=1)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted family")
    ax.set_ylabel("True family")
    ax.set_title("k-NN on the test set (share of each true family)")

    # Display the percentage and sample count in each cell
    for i in range(len(labels)):
        for j in range(len(labels)):
            colour = "white" if shares[i, j] > 0.5 else "black"
            ax.text(
                j, i,
                f"{shares[i, j]:.1%}\n({counts[i, j]:,})",
                ha="center", va="center",
                fontsize=8, color=colour,
            )

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)  # Create output folder
    fig.savefig(path, dpi=150)  # Save the graph
    plt.close(fig)  # Release memory
