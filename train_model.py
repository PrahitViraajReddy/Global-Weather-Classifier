"""Train and evaluate a reproducible temperature-category classifier.

Temperature classes are dataset-relative quantiles, not universal meteorological
categories. The test set is held out before quantile thresholds are fitted.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

DATA_PATH = Path("GlobalWeatherRepository.csv")
MODEL_PATH = Path("weather_model.pkl")
RANDOM_STATE = 42
FEATURE_COLUMNS = [
    "longitude", "latitude", "humidity", "cloud", "precip_in", "wind_kph",
    "visibility_km", "uv_index", "gust_mph", "pressure_mb", "air_quality_Ozone",
]
TARGET_COLUMN = "temperature_celsius"


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {path.resolve()}. "
            "Download the Git LFS dataset with 'git lfs pull' or place the CSV "
            "in the project root."
        )
    # Git LFS pointer files are tiny text files, not usable CSV datasets.
    with path.open("rb") as handle:
        prefix = handle.read(200)
    if b"git-lfs.github.com/spec/v1" in prefix:
        raise RuntimeError(
            f"{path} is a Git LFS pointer, not the actual dataset. "
            "Install Git LFS and run 'git lfs pull'."
        )
    df = pd.read_csv(path)
    required = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    df = df[required].copy()
    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[TARGET_COLUMN])
    df = df.drop_duplicates()
    if len(df) < 50:
        raise ValueError(f"Need at least 50 valid rows to train; found {len(df)}.")
    return df


def make_quantile_labels(train_y: pd.Series, test_y: pd.Series):
    """Fit class boundaries on training targets only, then apply to held-out targets."""
    train_bins = pd.qcut(train_y, q=5, retbins=True, duplicates="drop")[1]
    if len(train_bins) < 3:
        raise ValueError("Target has too few distinct values to form temperature classes.")
    inner_edges = train_bins[1:-1]
    # Infinite outer edges ensure test observations outside train range remain labelable.
    edges = np.concatenate(([-np.inf], inner_edges, [np.inf]))
    n_classes = len(edges) - 1
    train_labels = pd.cut(train_y, bins=edges, labels=False, include_lowest=True)
    test_labels = pd.cut(test_y, bins=edges, labels=False, include_lowest=True)
    labels = [f"Temperature quantile {i + 1}" for i in range(n_classes)]
    return train_labels.astype(int), test_labels.astype(int), edges.tolist(), labels


def main() -> None:
    df = load_data()
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()

    X_train, X_test, y_train_raw, y_test_raw = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE
    )
    y_train, y_test, bin_edges, class_labels = make_quantile_labels(
        y_train_raw, y_test_raw
    )
    # Fit imputation values only on training features.
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline

    model = make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)
    model_classes = model.named_steps["randomforestclassifier"].classes_
    ordered_labels = [class_labels[int(i)] for i in model_classes]

    print(f"Rows after cleaning: {len(df):,}")
    print(f"Training rows: {len(X_train):,}; held-out test rows: {len(X_test):,}")
    print(f"Classes: {class_labels}")
    print(f"Balanced accuracy: {balanced_accuracy_score(y_test, predictions):.4f}")
    print("Confusion matrix (true rows, predicted columns):")
    print(confusion_matrix(y_test, predictions, labels=model_classes))
    print("Classification report:")
    print(classification_report(
        y_test, predictions, labels=model_classes,
        target_names=ordered_labels, zero_division=0
    ))
    if len(model_classes) > 1 and len(np.unique(y_test)) > 1:
        print(f"Macro ROC AUC (OVR): {roc_auc_score(y_test, probabilities, labels=model_classes, multi_class='ovr', average='macro'):.4f}")
    else:
        print("Macro ROC AUC unavailable: the held-out set contains fewer than two classes.")

    joblib.dump({
        "model": model,
        "feature_columns": FEATURE_COLUMNS,
        "class_labels": class_labels,
        "temperature_bin_edges": bin_edges,
        "target_column": TARGET_COLUMN,
        "random_state": RANDOM_STATE,
    }, MODEL_PATH)
    print(f"Saved model bundle to {MODEL_PATH.resolve()}")


if __name__ == "__main__":
    main()
