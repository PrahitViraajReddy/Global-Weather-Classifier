# 🌦️ Global Weather Analytics & Classification

An end-to-end **weather analytics and machine learning project** built with Python, Streamlit, and Power BI.

The project analyzes **116,933 global weather records**, performs data exploration and preprocessing, classifies temperature ranges with a Random Forest model, and presents the results through interactive analytics dashboards.

> **Focus:** Data Analysis • EDA • Data Quality • Machine Learning • Power BI • Streamlit

---

## 🚀 Live Application

**[Open the Streamlit App](https://prahitviraajreddy-global-weather-classifier-app-vnr7xd.streamlit.app/)**

Enter weather conditions such as humidity, cloud cover, precipitation, wind speed, visibility, UV index, pressure, ozone, and location coordinates to obtain a predicted temperature category.

---

## 📊 Dashboard Preview

### Executive Analytics Dashboard

![Executive Dashboard](Analysis/Images/Executive%20Dashboard.png)

### ML Prediction Analysis

![ML Model Prediction](Analysis/Images/ML%20Model%20Prediction.png)

The repository also contains the complete **Power BI (.pbix)** dashboard under `Analysis/`.

---

## 🎯 Project Objective

The goal is to turn raw global weather observations into useful analytical outputs:

1. Understand the structure and quality of the weather data.
2. Explore relationships and trends across meteorological variables.
3. Prepare the data for temperature-category classification.
4. Train and evaluate a Random Forest classifier.
5. Connect model predictions with interactive BI analysis.
6. Provide a deployed interface for prediction.

---

## 📦 Dataset

- **Records:** 116,933
- **Domain:** Global weather and environmental observations
- **Target:** Temperature category
- **Model inputs include:** latitude, longitude, humidity, cloud cover, precipitation, wind speed, visibility, UV index, wind gust, pressure, and ozone.

The target temperature is divided into **five quantile-based categories** for multi-class classification.

---

## 🔎 Data Analysis & Preprocessing

The project includes:

- Descriptive statistics
- Missing-value checks
- Duplicate checks
- Distribution analysis
- Grouped trend analysis
- Temperature categorization
- Feature selection
- Train/test splitting
- Classification performance evaluation

### Data Quality Work in Power BI

The BI layer also documents data-quality issues discovered during analysis, including:

- Multilingual country-name duplicates that could split the same country across visuals.
- Extreme wind-speed observations requiring investigation and treatment.

Rather than only visualizing the data, these issues were investigated as part of the analytics workflow.

---

## 🤖 Machine Learning

### Model

**Random Forest Classifier**

### Evaluation

| Metric | Result |
|---|---:|
| Balanced Accuracy | ~0.71 |
| Macro AUC | ~0.96 |

The evaluation includes:

- Classification report
- Precision / Recall / F1
- Confusion matrix
- Multi-class ROC curves
- One-vs-rest AUC analysis

The model is used as one component of the wider analytics workflow rather than treating prediction as the only project output.

---

## 📈 Power BI Analytics

A dedicated Power BI dashboard is included in `Analysis/weather.pbix`.

### Dashboard coverage

- Executive Overview
- Trends Over Time
- Temperature
- Humidity
- Air Quality
- UV Index
- Wind Speed
- Precipitation
- Visibility
- Cloud Cover
- ML Prediction Analysis

The dashboard uses **DAX measures**, geographic visuals, a date table, and model-prediction analysis.

### Python + Power BI

The deployed Streamlit application provides the live interactive experience, while the Power BI file provides the deeper BI/reporting layer.

This keeps the live application independent of Power BI embedding requirements while still making the complete BI work available in the repository.

---

## 🧰 Tech Stack

**Programming & Analysis**
- Python
- Pandas
- NumPy

**Machine Learning**
- Scikit-learn
- Random Forest
- Classification metrics
- ROC/AUC analysis

**Visualization**
- Matplotlib
- Seaborn

**Business Intelligence**
- Power BI
- DAX

**Application**
- Streamlit

---

## 🗂️ Repository Structure

```text
Global-Weather-Classifier/
├── Analysis/
│   ├── Images/
│   ├── README.md
│   └── weather.pbix
├── GlobalWeatherRepository.csv
├── app.py
├── train_model.py
├── weather.py
├── weather_model.pkl
└── requirements.txt
```

---

## ▶️ Run Locally

Clone the repository and install the dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

To retrain the model:

```bash
python train_model.py
```

---

## 💡 What This Project Demonstrates

This project brings together an end-to-end workflow:

**Raw Data → Data Quality → EDA → Feature Preparation → ML Classification → Model Evaluation → Power BI Analytics → Streamlit Deployment**

It demonstrates practical experience with both **data analysis and predictive modelling**, with the final outputs designed to be explored rather than remaining only inside a notebook.

---

## 📌 Key Takeaways

- Worked with a large global weather dataset.
- Performed data-quality investigation and exploratory analysis.
- Built and evaluated a multi-class Random Forest classifier.
- Achieved ~0.71 balanced accuracy and ~0.96 macro AUC in the documented evaluation.
- Built a multi-page Power BI analytics dashboard with DAX and geographic analysis.
- Integrated model predictions into the BI layer.
- Deployed an interactive Streamlit prediction application.

---

## 👤 Author

**Madupu Prahit Viraaj Reddy**

[GitHub](https://github.com/PrahitViraajReddy) • [LinkedIn](https://www.linkedin.com/in/prahit-viraaj-reddy-madupu-5169332ba/)
