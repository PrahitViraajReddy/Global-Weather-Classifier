"""Train and evaluate the Global Weather temperature classifier.

This script performs the same core workflow used by the project:
1. Load the weather dataset.
2. Select model features and create five temperature categories.
3. Train a Random Forest classifier.
4. Save the trained model for the Streamlit app.
5. Print evaluation metrics and display one-vs-rest ROC curves.
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    auc,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize


DATA_PATH = Path("GlobalWeatherRepository.csv")
MODEL_PATH = Path("weather_model.pkl")
RANDOM_STATE = 42

FEATURE_COLUMNS = [
    "longitude",
    "latitude",
    "humidity",
    "cloud",
    "precip_in",
    "wind_kph",
    "visibility_km",
    "uv_index",
    "gust_mph",
    "pressure_mb",
    "air_quality_Ozone",
]

CATEGORY_NAMES = ["Freezing", "Cold", "Moderate", "Warm", "Hot"]


def load_data(path: Path) -> pd.DataFrame:
    """Load the weather dataset and validate the required columns."""
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}. "
            "Place GlobalWeatherRepository.csv in the project root."
        )

    df = pd.read_csv(path)
    required_columns = FEATURE_COLUMNS + ["temperature_celsius"]
    missing_columns = sorted(set(required_columns) - set(df.columns))

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    return df


def prepare_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Prepare model features and five quantile-based temperature classes."""
    X = df[FEATURE_COLUMNS].copy()

    y = pd.qcut(
        df["temperature_celsius"],
        q=5,
        labels=False,
        duplicates="drop",
    )

    if y.nunique() != len(CATEGORY_NAMES):
        raise ValueError(
            f"Expected {len(CATEGORY_NAMES)} temperature classes, "
            f"but found {y.nunique()}."
        )

    return X, y


def train_classifier(
    X: pd.DataFrame, y: pd.Series
) -> tuple[RandomForestClassifier, pd.Series, pd.Series]:
    """Split the data and train the Random Forest classifier."""
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
    )

    model = RandomForestClassifier(random_state=RANDOM_STATE)
    model.fit(X_train, y_train)

    return model, y_test, X_test


def evaluate_model(
    model: RandomForestClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> None:
    """Print classification metrics and display one-vs-rest ROC curves."""
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)

    balanced_accuracy = balanced_accuracy_score(y_test, predictions)
    macro_auc = roc_auc_score(
        y_test,
        probabilities,
        average="macro",
        multi_class="ovo",
    )

    print(f"Balanced accuracy: {balanced_accuracy:.4f}")
    print(f"Macro ROC AUC: {macro_auc:.4f}")
    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, predictions))
    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=CATEGORY_NAMES,
        )
    )

    y_test_binary = label_binarize(
        y_test,
        classes=np.arange(len(CATEGORY_NAMES)),
    )

    plt.figure(figsize=(8, 6))

    for class_index, category in enumerate(CATEGORY_NAMES):
        fpr, tpr, _ = roc_curve(
            y_test_binary[:, class_index],
            probabilities[:, class_index],
        )
        class_auc = auc(fpr, tpr)
        plt.plot(
            fpr,
            tpr,
            linewidth=2,
            label=f"{category} (AUC = {class_auc:.3f})",
        )

    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("One-vs-Rest ROC Curves")
    plt.legend()
    plt.tight_layout()
    plt.show()


def main() -> None:
    """Run the complete training and evaluation workflow."""
    print("Loading weather dataset...")
    df = load_data(DATA_PATH)
    print(f"Loaded {len(df):,} records.")

    X, y = prepare_data(df)
    model, y_test, X_test = train_classifier(X, y)

    joblib.dump(model, MODEL_PATH)
    print(f"Saved trained model to {MODEL_PATH}")

    evaluate_model(model, X_test, y_test)


if __name__ == "__main__":
    main()
