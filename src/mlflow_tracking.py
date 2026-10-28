"""Shared configuration for the project's local MLflow tracking store."""

import os
import sqlite3
from urllib.parse import unquote, urlparse
from urllib.request import url2pathname

import mlflow

from src import config


def _local_path(location):
    """Turn an artifact location into a local folder path, or None if it is remote."""
    parsed = urlparse(str(location))
    if parsed.scheme == "file":
        return url2pathname(unquote(parsed.path))
    if len(parsed.scheme) <= 1:
        return str(location)
    return None


def _is_usable(location):
    """Check that the artifact folder can be created on this computer."""
    path = _local_path(location)
    if path is None:
        return True
    try:
        os.makedirs(path, exist_ok=True)
        return True
    except OSError:
        return False


def _repoint_experiment(experiment_id):
    """Point an existing experiment at this project's artifact folder."""
    prefix = "sqlite:///"
    uri = str(config.MLFLOW_TRACKING_URI)
    if not uri.startswith(prefix):
        return False

    connection = sqlite3.connect(uri[len(prefix):])
    try:
        connection.execute(
            "UPDATE experiments SET artifact_location = ? WHERE experiment_id = ?",
            (str(config.MLFLOW_ARTIFACT_ROOT), experiment_id),
        )
        connection.commit()
    finally:
        connection.close()
    return True


def configure_tracking() -> None:

    # Tell MLflow where to store experiment tracking information
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)

    # The artifact folder from config must work on this computer
    if not _is_usable(config.MLFLOW_ARTIFACT_ROOT):
        raise RuntimeError(
            f"Cannot create the MLflow artifact folder {config.MLFLOW_ARTIFACT_ROOT}. "
            "Check MLFLOW_ARTIFACT_ROOT in src/config.py."
        )

    experiment = mlflow.get_experiment_by_name(config.MLFLOW_EXPERIMENT)

    if experiment is None:
        # Create the experiment if it doesn't already exist
        mlflow.create_experiment(
            config.MLFLOW_EXPERIMENT,
            artifact_location=config.MLFLOW_ARTIFACT_ROOT,
        )
    elif not _is_usable(experiment.artifact_location):
        # The experiment was created on another computer, so fix its saved folder
        if _repoint_experiment(experiment.experiment_id):
            print(f"Updated MLflow artifact folder to {config.MLFLOW_ARTIFACT_ROOT}")
        else:
            print("Warning: the MLflow artifact folder could not be updated.")

    # Select this experiment so subsequent runs are recorded under it
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT)