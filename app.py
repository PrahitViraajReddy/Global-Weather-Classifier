"""Streamlit interface for the Global Weather temperature classifier."""
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = Path("weather_model.pkl")
FEATURE_COLUMNS = [
    "longitude", "latitude", "humidity", "cloud", "precip_in", "wind_kph",
    "visibility_km", "uv_index", "gust_mph", "pressure_mb", "air_quality_Ozone",
]
LEGACY_CLASS_LABELS = ["Freezing", "Cold", "Moderate", "Warm", "Hot"]

st.set_page_config(page_title="Global Weather Analytics", page_icon="🌦️", layout="wide")


@st.cache_resource
def load_model_bundle():
    """Load the current bundle or adapt the repository's legacy raw model."""
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            "weather_model.pkl was not found. Ensure the model file is included in the deployment."
        )

    artifact = joblib.load(MODEL_PATH)

    # Older versions saved the sklearn model directly, rather than a metadata bundle.
    if not isinstance(artifact, dict) or "model" not in artifact:
        model = artifact.get("model") if isinstance(artifact, dict) else artifact
        if not hasattr(model, "predict"):
            raise ValueError("weather_model.pkl is not a supported scikit-learn model. Retrain it with train_model.py.")
        model_features = list(getattr(model, "feature_names_in_", FEATURE_COLUMNS))
        if model_features != FEATURE_COLUMNS:
            raise ValueError(
                "The saved legacy model uses a different feature schema. Retrain it with train_model.py."
            )
        return {
            "model": model,
            "feature_columns": FEATURE_COLUMNS,
            "class_labels": LEGACY_CLASS_LABELS,
            "legacy_model": True,
        }

    feature_columns = artifact.get("feature_columns")
    class_labels = artifact.get("class_labels")
    if feature_columns != FEATURE_COLUMNS or not isinstance(class_labels, list) or not class_labels:
        raise ValueError(
            "The saved model bundle has missing or incompatible feature/class metadata. "
            "Retrain it with train_model.py."
        )
    return {**artifact, "legacy_model": False}


def main() -> None:
    st.title("🌦️ Global Weather Analytics")
    st.caption("Classifies temperature relative to the training dataset; this is not a forecast or official weather advice.")
    try:
        bundle = load_model_bundle()
    except (FileNotFoundError, ValueError, OSError) as exc:
        st.error(str(exc))
        st.stop()

    model = bundle["model"]
    feature_columns = bundle["feature_columns"]
    class_labels = bundle["class_labels"]
    if bundle["legacy_model"]:
        st.warning(
            "A legacy model was loaded successfully. Its original temperature-bin boundaries were not saved, "
            "so the displayed category names are the legacy labels and should be treated as approximate. "
            "For fully reproducible category metadata, retrain with train_model.py."
        )

    st.divider()
    st.subheader("Temperature Category Prediction")
    with st.form("prediction_form"):
        loc, atmosphere, conditions = st.columns(3)
        with loc:
            st.markdown("**Location**")
            longitude = st.number_input("Longitude", min_value=-180.0, max_value=180.0, value=0.0, step=0.1)
            latitude = st.number_input("Latitude", min_value=-90.0, max_value=90.0, value=0.0, step=0.1)
        with atmosphere:
            st.markdown("**Atmospheric Conditions**")
            humidity = st.slider("Humidity (%)", 0, 100, 50)
            cloud = st.slider("Cloud Cover (%)", 0, 100, 20)
            pressure = st.number_input("Pressure (mb)", min_value=800.0, max_value=1100.0, value=1010.0, step=1.0)
            uv = st.number_input("UV Index", min_value=0.0, max_value=15.0, value=5.0, step=0.1)
        with conditions:
            st.markdown("**Weather Measurements**")
            precip = st.number_input("Precipitation (inches)", min_value=0.0, max_value=50.0, value=0.0, step=0.1)
            wind = st.number_input("Wind Speed (kph)", min_value=0.0, max_value=200.0, value=10.0, step=0.1)
            visibility = st.number_input("Visibility (km)", min_value=0.0, max_value=50.0, value=10.0, step=0.1)
            gust = st.number_input("Wind Gust (mph)", min_value=0.0, max_value=200.0, value=10.0, step=0.1)
            ozone = st.number_input("Ozone Level", min_value=0.0, max_value=500.0, value=50.0, step=1.0)
        submitted = st.form_submit_button("Predict Temperature Category", type="primary", use_container_width=True)

    if submitted:
        features = pd.DataFrame([{
            "longitude": longitude, "latitude": latitude, "humidity": humidity,
            "cloud": cloud, "precip_in": precip, "wind_kph": wind,
            "visibility_km": visibility, "uv_index": uv, "gust_mph": gust,
            "pressure_mb": pressure, "air_quality_Ozone": ozone,
        }], columns=feature_columns)
        prediction = int(model.predict(features)[0])
        model_classes = list(getattr(model, "classes_", []))
        if hasattr(model, "named_steps"):
            final_estimator = list(model.named_steps.values())[-1]
            model_classes = list(getattr(final_estimator, "classes_", model_classes))
        if prediction < 0 or prediction >= len(class_labels):
            st.error("The model returned a class outside the configured label range. Retrain the model bundle.")
            return

        category = class_labels[prediction]
        st.divider()
        st.subheader("Prediction Result")
        result_col, details_col = st.columns([1, 2])
        with result_col:
            st.metric("Temperature Category", category)
        with details_col:
            st.info(f"Predicted **{category}**. Temperature categories are dataset-relative, not fixed temperature ranges.")

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(features)[0]
            probability_labels = [class_labels[int(c)] for c in model_classes] if model_classes else class_labels[:len(probabilities)]
            probability_df = pd.DataFrame({"Model probability": probabilities}, index=probability_labels)
            st.write("### Class probabilities")
            st.bar_chart(probability_df)
        with st.expander("View submitted model inputs"):
            st.dataframe(features.T.rename(columns={0: "Value"}), use_container_width=True)

    st.divider()
    st.caption("Workflow: Data validation → Temperature categories → Train/test evaluation → Random Forest → Streamlit prediction")


if __name__ == "__main__":
    main()
