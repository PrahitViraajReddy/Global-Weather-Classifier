# 🌦️ Global Weather Classifier

**An end-to-end weather analytics and machine-learning project combining data exploration, temperature-category classification, and Power BI reporting.**

<p>
  <a href="https://prahitviraajreddy-global-weather-classifier-app-vnr7xd.streamlit.app/"><img alt="Live App" src="https://img.shields.io/badge/Live%20App-Open-2ea44f?style=flat-square"></a>
  <img alt="Python" src="https://img.shields.io/badge/Python-3.x-3776AB?style=flat-square&logo=python&logoColor=white">
  <img alt="Machine Learning" src="https://img.shields.io/badge/ML-Random%20Forest-orange?style=flat-square">
  <img alt="Power BI" src="https://img.shields.io/badge/Analytics-Power%20BI-F2C811?style=flat-square&logo=powerbi&logoColor=black">
  <img alt="Streamlit" src="https://img.shields.io/badge/App-Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white">
</p>

**[Open the live application](https://prahitviraajreddy-global-weather-classifier-app-vnr7xd.streamlit.app/)** · **[View the source code](https://github.com/PrahitViraajReddy/Global-Weather-Classifier)**

---

## What is Global Weather Classifier?

Global Weather Classifier is a data analytics and machine-learning project built around global weather observations. It combines data-quality checks, exploratory analysis, a Random Forest classification workflow, an interactive Streamlit interface, and a Power BI dashboard.

The application classifies supplied weather measurements into temperature categories learned from the training data. **It is a classifier, not a weather-forecasting service**, and the prediction interface does not fetch live weather observations.

## Features

- **Weather data analysis** — explore distributions, missing values, duplicates, and relationships between weather variables.
- **Data-quality investigation** — examine issues such as multilingual country-name duplicates and extreme wind-speed observations.
- **Temperature-category classification** — predict a temperature quantile category from supplied weather and location inputs.
- **Model evaluation** — review classification reports, balanced accuracy, confusion matrices, and ROC/AUC analysis where defined.
- **Power BI reporting** — explore weather patterns and model-prediction analysis in a dedicated dashboard.
- **Interactive web application** — enter weather conditions and receive a model prediction through Streamlit.

## How it works

```text
Weather observations
        │
        ▼
Data validation and exploration
        │
        ▼
Feature preparation and target construction
        │
        ▼
Train/test split and Random Forest training
        │
        ▼
Model evaluation and saved model artifact
        │
        ├──────────────► Streamlit prediction app
        │
        └──────────────► Power BI analytics
```

The training workflow validates the input data, constructs quantile-based target classes, fits preprocessing and the classifier using training data, evaluates predictions on held-out data, and saves the fitted model for the application.

## Dataset

The project documentation describes a dataset of **116,933 global weather records**.

- **Domain:** Weather and environmental observations
- **Prediction target:** Temperature category
- **Target construction:** Five quantile-based categories
- **Input features:** Latitude, longitude, humidity, cloud cover, precipitation, wind speed, wind gust, visibility, UV index, pressure, and ozone

Quantile labels represent relative positions within the training data. They should not be interpreted as universal meteorological labels such as “freezing” or “hot.”

The repository's `GlobalWeatherRepository.csv` is tracked with **Git LFS**. Use Git LFS to retrieve the actual dataset when cloning the repository.

## Machine learning and evaluation

**Model:** Random Forest Classifier

The documented evaluation reports approximately:

| Metric | Result |
|---|---:|
| Balanced accuracy | 0.71 |
| Macro ROC AUC | 0.96 |

The workflow also reports precision, recall, F1-score, a confusion matrix, and multi-class ROC/AUC analysis when the metric is defined.

These values describe the documented evaluation and may vary if the dataset, split, features, or training procedure changes. The evaluation uses a random row split; repeated observations from the same locations or time periods may warrant a more demanding location- or time-based holdout.

## Power BI analytics

The Power BI report is documented at `Analysis/weather.pbix` and covers:

- Executive overview
- Trends over time
- Temperature and humidity
- Air quality and UV index
- Wind speed and precipitation
- Visibility and cloud cover
- Machine-learning prediction analysis

The report uses DAX measures and geographic and time-based analysis to complement the prediction interface.

## Technology stack

| Layer | Technologies |
|---|---|
| Language | Python |
| Data analysis | Pandas, NumPy |
| Machine learning | scikit-learn, Random Forest |
| Evaluation | Classification metrics, ROC/AUC |
| Visualization | Matplotlib, Seaborn |
| Business intelligence | Power BI, DAX |
| Web interface | Streamlit |
| Dataset storage | Git LFS |

## Run locally

### 1. Clone the repository and retrieve the dataset

Install Git LFS first, then run:

```bash
git lfs install
git clone https://github.com/PrahitViraajReddy/Global-Weather-Classifier.git
cd Global-Weather-Classifier
git lfs pull
```

### 2. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 3. Train the model

```bash
python train_model.py
```

Training creates or refreshes `weather_model.pkl`. If the dataset is unavailable, training should stop with a clear error rather than treating the Git LFS pointer as CSV data.

### 4. Launch the application

```bash
streamlit run app.py
```

### 5. Run tests

```bash
python -m pytest -q
```

Retrain the model after changing the dataset, feature set, target construction, or training procedure.

## Repository structure

```text
Global-Weather-Classifier/
├── Analysis/
│   ├── Images/
│   └── weather.pbix
├── GlobalWeatherRepository.csv
├── app.py
├── train_model.py
├── weather.py
├── weather_model.pkl
├── requirements.txt
└── README.md
```

## Engineering highlights

- End-to-end data analysis and machine-learning workflow
- Data validation and preprocessing
- Training-only estimation of quantile boundaries and preprocessing
- Held-out model evaluation with classification metrics
- Saved model artifacts reused by the prediction application
- Interactive Streamlit deployment
- Power BI reporting with DAX and geographic analysis

## Links

- **Live application:** https://prahitviraajreddy-global-weather-classifier-app-vnr7xd.streamlit.app/
- **GitHub repository:** https://github.com/PrahitViraajReddy/Global-Weather-Classifier
- **Author:** [Madupu Prahit Viraaj Reddy](https://github.com/PrahitViraajReddy)
- **LinkedIn:** [Prahit Viraaj Reddy](https://www.linkedin.com/in/prahit-viraaj-reddy-madupu-5169332ba/)

---

<div align="center">

Built with Python, scikit-learn, Streamlit, and Power BI.

</div>
