## 📊 Power BI Analytics Dashboard

In addition to the ML classifier above, this project includes a full Power BI dashboard built on the same [Global Weather Repository](https://www.kaggle.com/datasets/nelgiriyewithana/global-weather-repository) dataset (116,933 records).

**File:** [`weather.pbix`](weather.pbix) — open in [Power BI Desktop](https://www.microsoft.com/en-us/power-platform/products/power-bi/downloads) (free) to explore interactively.

### What's inside

- **10 report pages** — Executive Overview, Trends Over Time, and dedicated pages for Temperature, Humidity, AQI, UV Index, WindSpeed, Precipitation, Visibility, and Cloud Cover
- **Data model** — fact table + a dedicated date table, with explicit DAX measures (averages, temperature categorization, prediction accuracy)
- **Geographic visuals** — country-level maps built with ArcGIS Maps for Power BI
- **ML integration** — a separate page surfaces the classifier's predictions (`Predicted Category`, `Correct Prediction`) alongside an `Accuracy` measure, connecting the model output back into the BI layer

### Data cleaning highlights

Two real data-quality issues were found and fixed during this build, rather than just visualized past:

- **Multilingual duplicate countries** — 10 rows had country names in Portuguese, German, Russian, Arabic, and Chinese (e.g. `Polônia`, `Südkorea`, `火鸡`) instead of English, silently splitting single countries into multiple entries on every chart. Identified via a non-ASCII character scan and merged back into their correct English names.
- **Wind-speed outliers** — a small number of records (5 of 116,933) had physically impossible wind speeds (one as high as 2963 kph). One case (Burundi, June 23) was cross-checked against real-world weather records and confirmed as a genuine storm event; the remaining extreme values were capped at the 99.9th percentile rather than dropped, to avoid discarding real rows over one unverified field.

### Screenshots

| Executive Overview | ML Model Prediction |
|---|---|
| ![Executive Dashboard](Images/Executive%20Dashboard.png) | ![ML Model Prediction](Images/ML%20Model%20Prediction.png) |

| Trends Over Time | Trends Over Time (2) |
|---|---|
| ![Trends Over Time 1](Images/Trends%20Over%20Time.png) | ![Trends Over Time 2](Images/Trends%20Over%20TIme(2).png) |

### Why Power BI *and* a native Python dashboard

The live Streamlit app above uses a Plotly-based analytics section (not an embedded Power BI report) — Power BI's public embedding requires either "Publish to Web" or an Azure app registration with tenant admin consent, neither of which is available on an institutional account. Rather than block the live demo on that, the same analysis was rebuilt natively in Python so the deployed app has zero external dependencies. The `.pbix` file is included here as a downloadable artifact for anyone who wants to see the full Power BI build — DAX measures, ArcGIS maps, and all.
