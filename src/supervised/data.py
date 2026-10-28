""" 2. Load processed data and select model input features."""
import json

import joblib
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif

from src import config

N_FEATURES = 20


def load_scaled():
    """Load the train/test data and apply the scaler saved during preparation."""
    train = pd.read_parquet(config.DATA_PROCESSED / "train.parquet")
    test = pd.read_parquet(config.DATA_PROCESSED / "test.parquet")
    scaler = joblib.load(config.DATA_PROCESSED / "scaler.joblib")
    cols = json.loads((config.DATA_PROCESSED / "feature_columns.json").read_text())

    X_train = scaler.transform(train[cols]).astype("float32")
    X_test = scaler.transform(test[cols]).astype("float32")
    return X_train, train["family"].to_numpy(), X_test, test["family"].to_numpy(), cols


def select_features(X_train, y_train, cols, k=N_FEATURES):
    """Select useful features using training data only to avoid test leakage."""
    k = min(k, X_train.shape[1])
    selector = SelectKBest(f_classif, k=k).fit(X_train, y_train)  # returns in the form of boolean true-selected , false-notSelected 
    chosen = [column for column, keep in zip(cols, selector.get_support()) if keep] # collects every selected features
    return selector, chosen
