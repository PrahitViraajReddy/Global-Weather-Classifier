# Global Weather Classifier

A Streamlit application that classifies observed weather inputs into **temperature quantiles learned from the training dataset**. It is a classifier, not a weather forecast, and it does not fetch live weather data.

## Live Streamlit app

[Open Global Weather Classifier](https://prahitviraajreddy-global-weather-classifier-app-vnr7xd.streamlit.app/)

## Run locally

Install Git LFS before cloning/pulling the dataset. The repository's `GlobalWeatherRepository.csv` is stored as a Git LFS object, so the small pointer file alone is not the actual CSV.

```bash
git lfs install
git clone https://github.com/PrahitViraajReddy/Global-Weather-Classifier.git
cd Global-Weather-Classifier
git lfs pull
python -m pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

If the dataset is not available, training stops with a clear error instead of attempting to parse the LFS pointer as CSV.

## Evaluation and methodology

The training script:
1. Validates the dataset schema, converts model fields to numeric values, drops invalid target rows and duplicate records.
2. Splits the raw temperature target into training and test portions **before** estimating temperature-quantile boundaries.
3. Learns class boundaries from the training target only and applies those boundaries to the held-out target.
4. Fits median imputation and a 300-tree Random Forest using training data only.
5. Reports held-out balanced accuracy, confusion matrix, per-class precision/recall/F1, and macro one-vs-rest ROC AUC when it is defined.
6. Saves the estimator and its feature names, quantile boundaries, and class labels in `weather_model.pkl`; the app uses these same labels.

Class names such as “Temperature quantile 1” are intentionally used instead of “Freezing” or “Hot”: quantile classes are relative to this dataset and do not represent universal meteorological temperature thresholds. The evaluation is a random row split; if the dataset contains repeated observations by location or time, a future stronger evaluation should hold out locations or later time periods separately.

## Features

The classifier uses longitude, latitude, humidity, cloud cover, precipitation, wind speed and gust, visibility, UV index, pressure, and ozone. All inputs should use the units shown in the app.

## Tests

```bash
python -m pytest -q
```

The saved model is a generated artifact. Retrain it after changing the data, features, target construction, or training procedure.
