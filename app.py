"""Streamlit interface for the Global Weather temperature classifier."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

MODEL_PATH = Path("weather_model.pkl")
FEATURE_COLUMNS = [
    "longitude", "latitude", "humidity", "cloud", "precip_in", "wind_kph",
    "visibility_km", "uv_index", "gust_mph", "pressure_mb", "air_quality_Ozone",
]
FALLBACK_LABELS = ["Freezing", "Cold", "Moderate", "Warm", "Hot"]

st.set_page_config(page_title="Global Weather Analytics", page_icon="🌦️", layout="wide")


@st.cache_resource
def load_model_bundle():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            "weather_model.pkl was not found. Download the dataset with 'git lfs pull', "
            "then run 'python train_model.py'."
        )
    artifact = joblib.load(MODEL_PATH)
    # Older model files stored only the estimator.
    if isinstance(artifact, dict) and "model" in artifact:
        return artifact
    return {"model": artifact, "feature_columns": FEATURE_COLUMNS, "class_labels": FALLBACK_LABELS}


def main() -> None:
    st.title("🌦️ Global Weather Analytics")
    st.caption("Classifies temperature relative to the training dataset; this is not a forecast or official weather advice.")
    try:
        bundle = load_model_bundle()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    model = bundle["model"]
    feature_columns = bundle.get("feature_columns", FEATURE_COLUMNS)
    class_labels = bundle.get("class_labels", FALLBACK_LABELS)

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
            st.error("The model returned a class outside the configured label range.")
            return
        category = class_labels[prediction]
        st.divider()
        st.subheader("Prediction Result")
        result_col, details_col = st.columns([1, 2])
        with result_col:
            st.metric("Temperature Category", category)
        with details_col:
            st.info(f"Predicted **{category}**. These labels are dataset-relative quantiles, not fixed temperature ranges.")

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(features)[0]
            # sklearn classes are the numeric labels used to train the classifier.
            probability_labels = [class_labels[int(c)] for c in model_classes] if model_classes else class_labels[:len(probabilities)]
            probability_df = pd.DataFrame({"Model probability": probabilities}, index=probability_labels)
            st.write("### Class probabilities")
            st.bar_chart(probability_df)
        with st.expander("View submitted model inputs"):
            st.dataframe(features.T.rename(columns={0: "Value"}), use_container_width=True)

    st.divider()
    st.caption("Workflow: Data validation → Training-only target bins → Train/test evaluation → Random Forest → Streamlit prediction")


if __name__ == "__main__":
    main()
