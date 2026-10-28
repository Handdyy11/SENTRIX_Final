# SENTRIX

SENTRIX is a network-traffic classification and threat-detection project built
on the CICIDS2017 intrusion-detection dataset. It provides a repeatable data
preparation and model-comparison pipeline, a Streamlit dashboard for exploring
traffic and trying predictions, and local MLflow tracking for experiments.

> **Project status:** Academic/demo software. Predictions and suggested
> responses are for exploration and evaluation; they are not a substitute for
> a production security monitoring system.

## What it does

- Cleans CICIDS2017 flow records in chunks to limit memory use.
- Groups individual traffic labels into five families: `Benign`, `DoS`,
  `PortScan`, `BruteForce`, and `Other`.
- Creates stratified train/test data, standardizes numeric features, and
  selects useful features for model training.
- Tunes k-NN on a validation split of the training data, then compares it with
  a decision tree, random forest, XGBoost, and a majority-class baseline.
- Reports accuracy, macro F1, per-family precision and recall, confusion
  matrices, and feature importance where supported.
- Tracks experiment parameters, metrics, models, and artifacts locally with
  MLflow.
- Provides a Streamlit dashboard with dataset overview, model results, live
  single-flow detection, traffic simulation, and an About page. Live Detection
  constrains edited feature values to the observed data range.

## Project layout

| Path | Purpose |
| --- | --- |
| `dashboard.py` | Streamlit application entry point |
| `src/config.py` | Shared project paths, family names, and processing settings |
| `src/preprocess/clean.py` | Clean, group, deduplicate, and sample raw CSV data |
| `src/preprocess/prepare.py` | Build train/test splits, fit the scaler, and save feature metadata |
| `src/supervised/models.py` | Entry point that orchestrates the model-comparison pipeline |
| `src/supervised/data.py` | Load scaled data and select model features |
| `src/supervised/training.py` | Tune k-NN and train the classifiers |
| `src/supervised/evaluation.py` | Calculate evaluation metrics and confusion matrices |
| `src/supervised/reporting.py` | Save model artifacts and results; print a summary |
| `src/supervised/tracking.py` | Record model runs, metrics, and artifacts in MLflow |
| `src/supervised/xgboost_model.py` | Adapt XGBoost to the project's string-valued labels |
| `data/raw/` | Raw CICIDS2017 CSV files used by preprocessing |
| `data/processed/` | Generated sample, train/test data, scaler, and model files |
| `results/` | Generated metrics, tuning table, and figures |
| `mlruns/` | Local MLflow database and run artifacts |
| `tests/` | Automated tests for preprocessing and model helpers |

Generated data, model files, results, and MLflow runs are local project outputs.
They may be ignored by version control; regenerate them with the pipeline if
they are absent.

## Requirements

- Windows and PowerShell (commands below use PowerShell syntax)
- 64-bit Python 3.12
- The raw CICIDS2017 CSV files in `data/raw/` to run preprocessing from scratch

The project has previously encountered incompatible compiled packages under
Python 3.14. Python 3.12 is the tested environment for this workspace. Keep the
virtual environment outside OneDrive and use its Python for installing
packages, running tests, and launching the dashboard. This avoids Windows
Application Control blocking compiled Python modules inside the synced project.

## Setup

Open PowerShell in the project root (the directory containing this README), then
create the environment under your local AppData directory and install all project
dependencies:

```powershell
$venv = Join-Path $env:LOCALAPPDATA 'venvs\sentrix312'
py -3.12 -m venv $venv
& "$venv\Scripts\python.exe" --version
& "$venv\Scripts\python.exe" -m pip install --upgrade pip
& "$venv\Scripts\python.exe" -m pip install -r requirements.txt
```

The version check should report Python 3.12. Streamlit and XGBoost are included
in `requirements.txt`. Using the virtual environment's Python executable
directly avoids accidentally launching the app with another Python installation.
Pandas is constrained below 3.0 to use the tested compatible release. The
previous virtual environment inside OneDrive could not load its compiled pandas
module under the Windows Application Control policy.

## Run the dashboard

If processed data and saved models already exist, start the application
directly:

```powershell
$venv = Join-Path $env:LOCALAPPDATA 'venvs\sentrix312'
& "$venv\Scripts\python.exe" -m streamlit run dashboard.py
```

Open the local URL printed by Streamlit, typically
<http://localhost:8501>. Keep the terminal open while using the dashboard.

The dashboard includes:

- **Overview:** dataset and traffic-family summaries.
- **Models:** comparison metrics and model results.
- **Live Detection:** select a flow, edit in-range features, and predict its
  family.
- **Simulation:** batch simulation, real-time stream, and CSV/Parquet upload
  workflows.
- **About Us:** project information.

Some dashboard pages depend on the generated processed data, results, and
saved model artifacts. Run the pipeline below if they are missing or you want
to rebuild them.

## Run the data and model pipeline

Run these commands from the project root, in order:

```powershell
$venv = Join-Path $env:LOCALAPPDATA 'venvs\sentrix312'
& "$venv\Scripts\python.exe" -m src.preprocess.clean
& "$venv\Scripts\python.exe" -m src.preprocess.prepare
& "$venv\Scripts\python.exe" -m src.supervised.models
```

The stages are:

1. **Clean:** reads CSV files from `data/raw/`, handles invalid/missing values
   and duplicates, maps labels to traffic families, and writes
   `data/processed/sample.parquet`.
2. **Prepare:** streams the sample into stratified train/test Parquet files,
   fits a `StandardScaler` using training data, and saves the feature list and
   preprocessing figures.
3. **Train and evaluate:** tunes k-NN using training data only, fits all
   classifiers, evaluates them on the held-out test data, and saves results
   and model artifacts.

Preprocessing can take time because it reads the raw data in chunks. Do not
run the model comparison until preparation has completed successfully.

## View MLflow experiments

After running a tracked pipeline stage, open a **second** PowerShell window in
the project root and start the MLflow UI with the same environment:

```powershell
$venv = Join-Path $env:LOCALAPPDATA 'venvs\sentrix312'
& "$venv\Scripts\python.exe" -m mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db
```

Open <http://127.0.0.1:5000>. Keep that terminal open while browsing runs.
Tracking data is stored locally under `mlruns/` and is specific to this
workspace.

## Run tests

From the project root:

```powershell
$venv = Join-Path $env:LOCALAPPDATA 'venvs\sentrix312'
& "$venv\Scripts\python.exe" -m pytest tests -q
```

## Outputs

Common generated files include:

- `data/processed/sample.parquet`, `train.parquet`, and `test.parquet`
- `data/processed/scaler.joblib` and `feature_columns.json`
- `data/processed/knn_model.joblib`, `tree_model.joblib`,
  `forest_model.joblib`, and `xgboost_model.joblib`
- `results/metrics_knn.json` and `results/knn_tuning.csv`
- `results/figures/confusion_matrix_knn.png` and feature histograms
- `mlruns/mlflow.db` and `mlruns/artifacts/`

## Notes

- The scaler and feature selection are fitted without using the held-out test
  data, to avoid test-data leakage.
- The dashboard's edited Live Detection inputs are bounded by the observed
  minimum and maximum values in the test dataset. Values outside those bounds
  are rejected before prediction.
- Do not load model artifacts from untrusted sources. Joblib model files can
  execute code when loaded.
- No license or dataset redistribution terms are specified in this repository.
  Check the CICIDS2017 dataset's own terms before redistributing its data.
