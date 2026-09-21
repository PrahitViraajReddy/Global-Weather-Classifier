"""Train the Random Forest model used by the Streamlit application."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


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


def main() -> None:
    """Load data, train the classifier, and save the model artifact."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}. "
            "Place GlobalWeatherRepository.csv in the project root."
        )

    df = pd.read_csv(DATA_PATH)

    required_columns = FEATURE_COLUMNS + ["temperature_celsius"]
    missing_columns = sorted(set(required_columns) - set(df.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    X = df[FEATURE_COLUMNS].copy()
    y = pd.qcut(
        df["temperature_celsius"],
        q=5,
        labels=False,
        duplicates="drop",
    )

    X_train, _, y_train, _ = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
    )

    model = RandomForestClassifier(random_state=RANDOM_STATE)
    model.fit(X_train, y_train)

    joblib.dump(model, MODEL_PATH)
    print(f"Saved trained model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
