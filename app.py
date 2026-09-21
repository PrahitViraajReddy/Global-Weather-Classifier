"""Streamlit interface for the Global Weather temperature classifier."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


MODEL_PATH = Path("weather_model.pkl")
CATEGORY_NAMES = ["Freezing", "Cold", "Moderate", "Warm", "Hot"]


st.set_page_config(
    page_title="Global Weather Analytics",
    page_icon="🌦️",
    layout="wide",
)


@st.cache_resource
def load_model():
    """Load the trained classifier once per Streamlit session."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "weather_model.pkl was not found. "
            "Run train_model.py first or deploy the trained model artifact."
        )
    return joblib.load(MODEL_PATH)


def main() -> None:
    """Render the weather prediction application."""
    st.title("🌦️ Global Weather Analytics")
    st.caption(
        "Explore weather conditions and predict a temperature category "
        "using the project's trained Random Forest classifier."
    )

    try:
        model = load_model()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    st.divider()

    st.subheader("Temperature Category Prediction")
    st.write(
        "Enter the weather conditions below. The model uses the same "
        "feature set used during training."
    )

    with st.form("prediction_form"):
        location_col, climate_col, environment_col = st.columns(3)

        with location_col:
            st.markdown("**Location**")
            longitude = st.number_input(
                "Longitude",
                min_value=-180.0,
                max_value=180.0,
                value=0.0,
                step=0.1,
            )
            latitude = st.number_input(
                "Latitude",
                min_value=-90.0,
                max_value=90.0,
                value=0.0,
                step=0.1,
            )

        with climate_col:
            st.markdown("**Atmospheric Conditions**")
            humidity = st.slider("Humidity (%)", 0, 100, 50)
            cloud = st.slider("Cloud Cover (%)", 0, 100, 20)
            pressure = st.number_input(
                "Pressure (mb)",
                min_value=800.0,
                max_value=1100.0,
                value=1010.0,
                step=1.0,
            )
            uv = st.number_input(
                "UV Index",
                min_value=0.0,
                max_value=15.0,
                value=5.0,
                step=0.1,
            )

        with environment_col:
            st.markdown("**Weather Measurements**")
            precip = st.number_input(
                "Precipitation (inches)",
                min_value=0.0,
                max_value=50.0,
                value=0.0,
                step=0.1,
            )
            wind_kph = st.number_input(
                "Wind Speed (kph)",
                min_value=0.0,
                max_value=200.0,
                value=10.0,
                step=0.1,
            )
            visibility = st.number_input(
                "Visibility (km)",
                min_value=0.0,
                max_value=50.0,
                value=10.0,
                step=0.1,
            )
            gust = st.number_input(
                "Wind Gust (mph)",
                min_value=0.0,
                max_value=200.0,
                value=10.0,
                step=0.1,
            )
            ozone = st.number_input(
                "Ozone Level",
                min_value=0.0,
                max_value=500.0,
                value=50.0,
                step=1.0,
            )

        submitted = st.form_submit_button(
            "Predict Temperature Category",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        features = pd.DataFrame(
            [
                [
                    longitude,
                    latitude,
                    humidity,
                    cloud,
                    precip,
                    wind_kph,
                    visibility,
                    uv,
                    gust,
                    pressure,
                    ozone,
                ]
            ],
            columns=[
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
            ],
        )

        prediction = int(model.predict(features)[0])
        category = CATEGORY_NAMES[prediction]

        st.divider()
        st.subheader("Prediction Result")

        result_col, details_col = st.columns([1, 2])

        with result_col:
            st.metric("Temperature Category", category)

        with details_col:
            st.success(
                f"The model classified the supplied conditions as **{category}**."
            )

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(features)[0]
            probability_df = pd.DataFrame(
                {
                    "Category": CATEGORY_NAMES,
                    "Model Probability": probabilities,
                }
            ).set_index("Category")

            st.write("### Class probabilities")
            st.bar_chart(probability_df)

        with st.expander("View submitted model inputs"):
            st.dataframe(
                features.T.rename(columns={0: "Value"}),
                use_container_width=True,
            )

    st.divider()
    st.caption(
        "Project workflow: Data Quality → EDA → Feature Preparation → "
        "Random Forest Classification → Evaluation → Power BI Analytics → "
        "Streamlit Deployment"
    )


if __name__ == "__main__":
    main()
