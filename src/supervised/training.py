
"""3. Tune and fit the supervised classifiers."""

import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight

from src import config
from src.supervised.xgboost_model import StringLabelXGBClassifier


# Hyperparameter options to test for KNN
K_CHOICES = [3, 5, 7, 9, 11, 13, 15]
WEIGHT_CHOICES = ["uniform", "distance"]

# Limit tuning to 40,000 rows to reduce computation time
TUNING_ROWS = 40_000


def choose_k(X, y, seed=config.SEED):
    """Find the best KNN hyperparameters using validation data."""

    # If the training dataset is too large, select a stratified subset
    if len(X) > TUNING_ROWS:
        X, _, y, _ = train_test_split(
            X,
            y,
            train_size=TUNING_ROWS,
            stratify=y,       # Preserve class proportions
            random_state=seed # Make the split reproducible
        )

    # Split data into fitting data (75%) and validation data (25%)
    # The validation data helps us compare different KNN settings
    X_fit, X_val, y_fit, y_val = train_test_split(
        X,
        y,
        test_size=0.25,
        stratify=y,
        random_state=seed
    )

    rows = []

    # Try every combination of k and voting strategy
    for weights in WEIGHT_CHOICES:
        for k in K_CHOICES:

            # Create KNN with the current hyperparameter combination
            model = KNeighborsClassifier(
                n_neighbors=k,
                weights=weights,
                n_jobs=-1
            )

            # Fit the model on fitting data
            model.fit(X_fit, y_fit)

            # Predict validation labels and measure performance
            # Macro F1 gives equal importance to each class
            score = f1_score(
                y_val,
                model.predict(X_val),
                average="macro"
            )

            # Save this combination and its score
            rows.append({
                "k": k,
                "weights": weights,
                "macro_f1": score
            })

    # Convert all results into a table
    table = pd.DataFrame(rows)

    # Sort by Macro F1 (highest first) and select the best combination
    best = table.sort_values(
        "macro_f1",
        ascending=False
    ).iloc[0]

    # Return the best settings and the complete results table
    return int(best["k"]), str(best["weights"]), table


def train_models(X_train, y_train, X_test, best_k, best_weights):
    """Train classifiers and generate predictions for test data."""

    print(
        "\nTraining the final k-NN on all training rows and scoring the test set "
        "(this can take a few minutes) ..."
    )

    # Train KNN on all training data using the settings selected earlier
    knn = KNeighborsClassifier(
        n_neighbors=best_k,
        weights=best_weights,
        n_jobs=-1
    )
    knn.fit(X_train, y_train)

    # Baseline: always predict the most frequent training class
    # Used to compare actual models against a simple strategy
    baseline = DummyClassifier(
        strategy="most_frequent"
    ).fit(X_train, y_train)

    # Train a Decision Tree with a maximum depth of 12
    tree = DecisionTreeClassifier(
        max_depth=12,
        random_state=config.SEED
    ).fit(X_train, y_train)

    print("\nTraining the random forest ...")

    # Train a Random Forest containing 200 decision trees
    forest = RandomForestClassifier(
        n_estimators=200,             # Number of trees
        max_depth=None,               # No explicit maximum depth
        min_samples_leaf=2,           # At least 2 samples per leaf
        class_weight="balanced_subsample",  # Adjust for class imbalance
        n_jobs=-1,                    # Use available CPU cores
        random_state=config.SEED
    ).fit(X_train, y_train)

    print("\nTraining XGBoost ...")

    # Configure the XGBoost classifier
    xgboost = StringLabelXGBClassifier(
        n_estimators=300,       # Number of boosting trees
        max_depth=8,             # Maximum depth of each tree
        learning_rate=0.1,       # Controls each boosting update
        subsample=0.9,           # Use 90% of rows per boosting round
        colsample_bytree=0.9,    # Use 90% of features per tree
        min_child_weight=2,
        n_jobs=-1,
        random_state=config.SEED
    )

    # Give more weight to samples from less frequent classes
    # so that the model pays more attention to them during training
    xgboost.fit(
        X_train,
        y_train,
        sample_weight=compute_sample_weight("balanced", y_train)
    )

    # Store all trained models in one dictionary
    estimators = {
        "knn": knn,
        "decision_tree": tree,
        "random_forest": forest,
        "xgboost": xgboost,
        "majority_baseline": baseline,
    }

    # Generate predictions for every test sample using each model
    predictions = {
        name: model.predict(X_test)
        for name, model in estimators.items()
    }

    # Return both trained models and their predictions for evaluation
    return estimators, predictions
