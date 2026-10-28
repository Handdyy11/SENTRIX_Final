
"""Shared configuration for the project's local MLflow tracking store."""

import mlflow
from src import config

def configure_tracking() -> None:

    # Tell MLflow where to store experiment tracking information
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)

    # Create the experiment if it doesn't already exist
    if mlflow.get_experiment_by_name(config.MLFLOW_EXPERIMENT) is None:
        mlflow.create_experiment(
            config.MLFLOW_EXPERIMENT,
            artifact_location=config.MLFLOW_ARTIFACT_ROOT,
        )

    # Select this experiment so subsequent runs are recorded under it
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT)
