import streamlit as st
import joblib
import numpy as np

model = joblib.load("weather_model.pkl")

st.title("🌦️ Weather Temperature Classifier")

st.write("Enter the weather conditions below to predict the temperature category.")

# Input fields with realistic ranges
longitude = st.number_input("Longitude", -180.0, 180.0, 0.0)
latitude = st.number_input("Latitude", -90.0, 90.0, 0.0)
humidity = st.slider("Humidity (%)", 0, 100, 50)
cloud = st.slider("Cloud Cover (%)", 0, 100, 20)
precip = st.number_input("Precipitation (inches)", 0.0, 50.0, 0.0)
wind_kph = st.number_input("Wind Speed (kph)", 0.0, 200.0, 10.0)
visibility = st.number_input("Visibility (km)", 0.0, 50.0, 10.0)
uv = st.number_input("UV Index", 0.0, 15.0, 5.0)
gust = st.number_input("Wind Gust (mph)", 0.0, 200.0, 10.0)
pressure = st.number_input("Pressure (mb)", 800.0, 1100.0, 1010.0)
ozone = st.number_input("Ozone Level", 0.0, 500.0, 50.0)

if st.button("Predict Temperature Category"):
    features = np.array([[longitude, latitude, humidity, cloud, precip,
                          wind_kph, visibility, uv, gust, pressure, ozone]])
    
    pred = model.predict(features)[0]
    categories = ["Freezing", "Cold", "Moderate", "Warm", "Hot"]

    st.success(f"Predicted Temperature Category: **{categories[pred]}**")
