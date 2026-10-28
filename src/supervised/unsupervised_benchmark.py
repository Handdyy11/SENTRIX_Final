"""Unsupervised models: Isolation Forest and K-Means on the 20 selected features."""

import json

import joblib
import matplotlib

matplotlib.use("Agg")  # Save plots without opening a window

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    average_precision_score,
    completeness_score,
    confusion_matrix,
    homogeneity_score,
    normalized_mutual_info_score,
    precision_recall_fscore_support,
    roc_auc_score,
    silhouette_score,
)

from src import config
from src.mlflow_tracking import configure_tracking
from src.supervised.data import load_scaled, select_features
from src.supervised.evaluation import evaluate

BENIGN_LABEL = "Benign"

# Isolation Forest settings
IFOREST_TREES = 200
IFOREST_MAX_SAMPLES = 1024

# K-Means settings
K_CHOICES = list(range(2, 13))
KMEANS_TUNING_ROWS = 40_000
SILHOUETTE_ROWS = 10_000


def to_native(obj):
    """Convert numpy types so they can be written to JSON."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Cannot convert {type(obj)} to JSON")


def fit_isolation_forest(X_train, y_train):
    """Fit Isolation Forest on all training rows without using labels."""

    # The only label-derived value: the share of attacks in the training data
    attack_share = float(np.mean(y_train != BENIGN_LABEL))
    contamination = min(max(attack_share, 0.001), 0.5)

    print(
        f"\nTraining Isolation Forest on {len(X_train):,} rows "
        f"(contamination = {contamination:.3f}) ..."
    )
    model = IsolationForest(
        n_estimators=IFOREST_TREES,
        max_samples=IFOREST_MAX_SAMPLES,
        contamination=contamination,
        n_jobs=-1,
        random_state=config.SEED,
    ).fit(X_train)

    return model, contamination


def evaluate_isolation_forest(model, X_test, y_test):
    """Treat every attack family as one anomaly class and score the test set."""

    is_attack = y_test != BENIGN_LABEL

    # Higher score means more anomalous
    scores = -model.score_samples(X_test)

    # predict() returns -1 for anomalies and 1 for normal rows
    flagged = model.predict(X_test) == -1

    precision, recall, f1, _ = precision_recall_fscore_support(
        is_attack, flagged, average="binary", zero_division=0
    )
    tn, fp, fn, tp = confusion_matrix(is_attack, flagged, labels=[False, True]).ravel()

    # Share of rows flagged as anomalies in each family
    # For Benign this is the false alarm rate, for attacks it is the detection rate
    flagged_share = {
        family: round(float(flagged[y_test == family].mean()), 4)
        for family in sorted(set(y_test))
    }

    results = {
        "roc_auc": round(float(roc_auc_score(is_attack, scores)), 4),
        "average_precision": round(float(average_precision_score(is_attack, scores)), 4),
        "accuracy": round(float(accuracy_score(is_attack, flagged)), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "confusion": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "flagged_share_per_family": flagged_share,
    }
    return results, scores


def plot_iforest_scores(scores, y_test, threshold, path):
    """Plot anomaly scores for benign and attack rows with the decision threshold."""

    is_attack = y_test != BENIGN_LABEL

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(scores[~is_attack], bins=60, alpha=0.6, density=True, label="Benign")
    ax.hist(scores[is_attack], bins=60, alpha=0.6, density=True, label="Attack")
    ax.axvline(threshold, color="black", linestyle="--", label="Decision threshold")
    ax.set_xlabel("Anomaly score (higher = more anomalous)")
    ax.set_ylabel("Density")
    ax.set_title("Isolation Forest scores on the test set")
    ax.legend()

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def choose_k(X_train):
    """Pick the number of clusters using silhouette score on a training subset."""

    rng = np.random.default_rng(config.SEED)
    if len(X_train) > KMEANS_TUNING_ROWS:
        idx = rng.choice(len(X_train), size=KMEANS_TUNING_ROWS, replace=False)
        X_tune = X_train[idx]
    else:
        X_tune = X_train

    rows = []
    for k in K_CHOICES:
        model = KMeans(n_clusters=k, n_init=5, random_state=config.SEED).fit(X_tune)

        # Silhouette is slow on large data, so it is computed on a sample
        score = silhouette_score(
            X_tune,
            model.labels_,
            sample_size=min(SILHOUETTE_ROWS, len(X_tune)),
            random_state=config.SEED,
        )
        rows.append({"k": k, "silhouette": score, "inertia": model.inertia_})
        print(f"  k={k:<3} silhouette={score:.4f}  inertia={model.inertia_:,.0f}")

    table = pd.DataFrame(rows)
    best_k = int(table.sort_values("silhouette", ascending=False).iloc[0]["k"])
    return best_k, table


def plot_k_selection(table, best_k, path):
    """Plot silhouette score and inertia against k."""

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(table["k"], table["silhouette"], marker="o")
    axes[0].axvline(best_k, color="red", linestyle="--", label=f"Best k = {best_k}")
    axes[0].set_xlabel("Number of clusters (k)")
    axes[0].set_ylabel("Silhouette score")
    axes[0].set_title("Silhouette score")
    axes[0].legend()

    axes[1].plot(table["k"], table["inertia"], marker="o")
    axes[1].set_xlabel("Number of clusters (k)")
    axes[1].set_ylabel("Inertia")
    axes[1].set_title("Elbow plot")

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fit_kmeans(X_train, best_k):
    """Fit the final K-Means on all training rows without using labels."""

    print(f"\nTraining K-Means with k={best_k} on {len(X_train):,} rows ...")
    return KMeans(n_clusters=best_k, n_init=10, random_state=config.SEED).fit(X_train)


def evaluate_kmeans(model, y_train, X_test, y_test):
    """Measure how well the clusters line up with the attack families."""

    # Name each cluster after the most common training family inside it
    train_table = pd.crosstab(model.labels_, y_train)
    cluster_to_family = {
        int(cluster): str(family)
        for cluster, family in train_table.idxmax(axis=1).items()
    }

    # Assign each test row to a cluster, then to that cluster's family
    test_clusters = model.predict(X_test)
    predicted_family = pd.Series(test_clusters).map(cluster_to_family).to_numpy()

    # Cluster composition on the test set (rows = clusters, columns = families)
    contingency = pd.crosstab(test_clusters, y_test).reindex(
        range(model.n_clusters), fill_value=0
    )
    purity = float(contingency.max(axis=1).sum() / contingency.to_numpy().sum())

    silhouette = silhouette_score(
        X_test,
        test_clusters,
        sample_size=min(SILHOUETTE_ROWS, len(X_test)),
        random_state=config.SEED,
    )

    results = {
        "k": int(model.n_clusters),
        "silhouette_test": round(float(silhouette), 4),
        "purity": round(purity, 4),
        "adjusted_rand_index": round(float(adjusted_rand_score(y_test, test_clusters)), 4),
        "normalized_mutual_info": round(
            float(normalized_mutual_info_score(y_test, test_clusters)), 4
        ),
        "homogeneity": round(float(homogeneity_score(y_test, test_clusters)), 4),
        "completeness": round(float(completeness_score(y_test, test_clusters)), 4),
        "cluster_to_family": cluster_to_family,
        "family_prediction": evaluate(y_test, predicted_family),
        "cluster_sizes_test": {
            int(c): int(n) for c, n in contingency.sum(axis=1).items()
        },
    }
    return results, contingency


def plot_cluster_heatmap(contingency, path):
    """Plot the family mix inside each cluster on the test set."""

    counts = contingency.to_numpy()
    shares = counts / counts.sum(axis=1, keepdims=True).clip(min=1)

    fig, ax = plt.subplots(figsize=(7, 0.5 * len(contingency) + 3))
    ax.imshow(shares, cmap="Blues", vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(len(contingency.columns)))
    ax.set_xticklabels(contingency.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(contingency.index)))
    ax.set_yticklabels([f"Cluster {c}" for c in contingency.index])
    ax.set_xlabel("True family")
    ax.set_title("K-Means clusters vs attack families (test set)")

    for i in range(counts.shape[0]):
        for j in range(counts.shape[1]):
            colour = "white" if shares[i, j] > 0.5 else "black"
            ax.text(
                j, i,
                f"{shares[i, j]:.0%}\n({counts[i, j]:,})",
                ha="center", va="center", fontsize=7, color=colour,
            )

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def print_summary(results):
    """Print a readable summary of both unsupervised models."""

    iforest = results["isolation_forest"]
    kmeans = results["kmeans"]

    print("\nIsolation Forest (attack vs benign) on the test set")
    print(f"  ROC AUC            {iforest['roc_auc']:.4f}")
    print(f"  Average precision  {iforest['average_precision']:.4f}")
    print(f"  Accuracy           {iforest['accuracy']:.4f}")
    print(f"  Precision          {iforest['precision']:.4f}")
    print(f"  Recall             {iforest['recall']:.4f}")
    print(f"  F1                 {iforest['f1']:.4f}")
    print("  Share flagged as anomaly per family")
    for family, share in iforest["flagged_share_per_family"].items():
        print(f"    {family:<11}{share:.3f}")

    print(f"\nK-Means with k={kmeans['k']} on the test set")
    print(f"  Silhouette         {kmeans['silhouette_test']:.4f}")
    print(f"  Purity             {kmeans['purity']:.4f}")
    print(f"  Adjusted Rand      {kmeans['adjusted_rand_index']:.4f}")
    print(f"  Normalized MI      {kmeans['normalized_mutual_info']:.4f}")
    print(f"  Homogeneity        {kmeans['homogeneity']:.4f}")
    print(f"  Completeness       {kmeans['completeness']:.4f}")
    print(
        f"  Family accuracy    {kmeans['family_prediction']['accuracy']:.4f}   "
        f"macro F1 {kmeans['family_prediction']['macro_f1']:.4f}"
    )
    print("  Cluster names (majority training family)")
    for cluster, family in kmeans["cluster_to_family"].items():
        print(f"    Cluster {cluster:<3}{family}")


def save_outputs(
    results, iforest, kmeans, selector, cols, table,
    contingency, scores, y_test, threshold,
):
    """Save metrics, plots, tables, and trained models."""

    config.RESULTS.mkdir(parents=True, exist_ok=True)
    figures = config.RESULTS / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    (config.RESULTS / "metrics_unsupervised.json").write_text(
        json.dumps(results, indent=2, default=to_native)
    )
    table.to_csv(config.RESULTS / "kmeans_tuning.csv", index=False)
    contingency.to_csv(config.RESULTS / "kmeans_cluster_family_test.csv")

    plot_iforest_scores(scores, y_test, threshold, figures / "iforest_score_distribution.png")
    plot_k_selection(table, results["kmeans"]["k"], figures / "kmeans_k_selection.png")
    plot_cluster_heatmap(contingency, figures / "kmeans_cluster_family_heatmap.png")

    # Same dictionary layout as the supervised models, plus extra details
    joblib.dump(
        {
            "model": iforest,
            "selector": selector,
            "features": cols,
            "threshold": threshold,
        },
        config.DATA_PROCESSED / "iforest_model.joblib",
    )
    joblib.dump(
        {
            "model": kmeans,
            "selector": selector,
            "features": cols,
            "cluster_to_family": results["kmeans"]["cluster_to_family"],
        },
        config.DATA_PROCESSED / "kmeans_model.joblib",
    )


def log_to_mlflow(results, chosen, cols, n_train, n_test, contamination, table):
    """Record parameters, metrics, and files in MLflow."""

    iforest = results["isolation_forest"]
    kmeans = results["kmeans"]

    mlflow.log_params({
        "random_seed": config.SEED,
        "training_rows": n_train,
        "testing_rows": n_test,
        "input_feature_count": len(cols),
        "selected_feature_count": len(chosen),
        "selected_features": ", ".join(chosen),
        "iforest_n_estimators": IFOREST_TREES,
        "iforest_max_samples": IFOREST_MAX_SAMPLES,
        "iforest_contamination": round(contamination, 4),
        "kmeans_candidate_k": ", ".join(map(str, K_CHOICES)),
        "kmeans_best_k": kmeans["k"],
        "kmeans_tuning_rows": KMEANS_TUNING_ROWS,
        "silhouette_sample_rows": SILHOUETTE_ROWS,
    })

    mlflow.log_metrics({
        "iforest_roc_auc": iforest["roc_auc"],
        "iforest_average_precision": iforest["average_precision"],
        "iforest_accuracy": iforest["accuracy"],
        "iforest_precision": iforest["precision"],
        "iforest_recall": iforest["recall"],
        "iforest_f1": iforest["f1"],
        "kmeans_silhouette_test": kmeans["silhouette_test"],
        "kmeans_purity": kmeans["purity"],
        "kmeans_adjusted_rand_index": kmeans["adjusted_rand_index"],
        "kmeans_normalized_mutual_info": kmeans["normalized_mutual_info"],
        "kmeans_homogeneity": kmeans["homogeneity"],
        "kmeans_completeness": kmeans["completeness"],
        "kmeans_family_accuracy": kmeans["family_prediction"]["accuracy"],
        "kmeans_family_macro_f1": kmeans["family_prediction"]["macro_f1"],
    })

    for family, share in iforest["flagged_share_per_family"].items():
        mlflow.log_metric(f"iforest_flagged_share_{family}", share)

    artifacts = (
        config.RESULTS / "metrics_unsupervised.json",
        config.RESULTS / "kmeans_tuning.csv",
        config.RESULTS / "kmeans_cluster_family_test.csv",
        config.RESULTS / "figures" / "iforest_score_distribution.png",
        config.RESULTS / "figures" / "kmeans_k_selection.png",
        config.RESULTS / "figures" / "kmeans_cluster_family_heatmap.png",
        config.DATA_PROCESSED / "iforest_model.joblib",
        config.DATA_PROCESSED / "kmeans_model.joblib",
    )
    for artifact in artifacts:
        mlflow.log_artifact(str(artifact), artifact_path="unsupervised-results")


def _run_models() -> None:

    # Load scaled training/test data, labels, and feature names
    X_train, y_train, X_test, y_test, cols = load_scaled()
    print(f"Train: {X_train.shape[0]:,} rows   Test: {X_test.shape[0]:,} rows")

    # Use the same 20 features as the supervised models
    selector, chosen = select_features(X_train, y_train, cols)
    X_train_selected = selector.transform(X_train)
    X_test_selected = selector.transform(X_test)
    print(f"Kept {len(chosen)} of {len(cols)} features")

    # Isolation Forest
    iforest, contamination = fit_isolation_forest(X_train_selected, y_train)
    iforest_results, scores = evaluate_isolation_forest(iforest, X_test_selected, y_test)
    threshold = float(-iforest.offset_)

    # K-Means
    print("\nChoosing k on a subset of the training data ...")
    best_k, table = choose_k(X_train_selected)
    print(f"\nBest: k={best_k}")
    kmeans = fit_kmeans(X_train_selected, best_k)
    kmeans_results, contingency = evaluate_kmeans(
        kmeans, y_train, X_test_selected, y_test
    )

    results = {
        "isolation_forest": iforest_results,
        "kmeans": kmeans_results,
        "tuning": table.round(4).to_dict("records"),
        "features_used": chosen,
    }

    save_outputs(
        results, iforest, kmeans, selector, cols, table,
        contingency, scores, y_test, threshold,
    )
    print_summary(results)

    log_to_mlflow(
        results, chosen, cols,
        len(y_train), len(y_test), contamination, table,
    )

    print(f"\nSaved {config.RESULTS / 'metrics_unsupervised.json'}")
    print(f"Saved {config.DATA_PROCESSED / 'iforest_model.joblib'}")
    print(f"Saved {config.DATA_PROCESSED / 'kmeans_model.joblib'}")


def main() -> None:
    # Configure MLflow, then create a parent run for this experiment
    configure_tracking()
    with mlflow.start_run(run_name="unsupervised-model-comparison"):
        _run_models()


# Run the pipeline when this file is executed directly
if __name__ == "__main__":
    main()