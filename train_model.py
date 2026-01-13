import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

df = pd.read_csv("GlobalWeatherRepository.csv")

X = df[['longitude','latitude','humidity','cloud','precip_in','wind_kph',
        'visibility_km','uv_index','gust_mph','pressure_mb','air_quality_Ozone']]

y = pd.cut(df['temperature_celsius'], q=5, labels=False)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

joblib.dump(model, "weather_model.pkl")
print("weather_model.pkl saved successfully!")
