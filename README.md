# Global Weather Classifier

<p align="center">
  <strong>A machine learning application for classifying global weather conditions into temperature categories.</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.14-blue?logo=python&logoColor=white" alt="Python 3.14">
  <img src="https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/Model-Random%20Forest-2E8B57" alt="Random Forest">
  <img src="https://img.shields.io/badge/Status-Active-success" alt="Project status">
</p>

## Overview

Global Weather Classifier is a Streamlit-based machine learning project that predicts a temperature category from weather and atmospheric measurements. It combines a trained Random Forest classifier with an interactive interface for entering conditions, viewing the prediction, and inspecting the model's class probabilities.

**The application classifies temperature ranges; it is not a weather-forecasting service and does not fetch live weather data.**

## Features

- **Temperature category prediction** using a trained Random Forest classifier.
- **Weather input form** for location coordinates, humidity, cloud cover, precipitation, wind, visibility, UV index, pressure, and ozone.
- **Class probability chart** to compare the model's output probabilities.
- **Input summary** so users can review the values submitted to the model.
- **Separate training script** to build and save the model artifact.
- **Streamlit interface** designed for interactive use.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python 3.14 | Application and model-training code |
| Streamlit | Interactive web interface |
| Pandas | Data loading and feature preparation |
| NumPy | Numerical operations |
| scikit-learn | Random Forest classification and data splitting |
| Joblib | Saving and loading the trained model |
| Matplotlib and Seaborn | Data visualization support |

## How It Works

1. **Prepare the data:** provide the training CSV expected by the training script.
2. **Create temperature categories:** the script divides the target temperature values into quantile-based classes.
3. **Train the classifier:** a Random Forest model is fitted using the selected weather features.
4. **Save the model:** the trained model is stored as `weather_model.pkl`.
5. **Make a prediction:** the Streamlit app loads the saved model and accepts weather measurements.
6. **Explore the result:** the app displays the predicted category, class probabilities when available, and submitted inputs.

## Temperature Categories

The interface displays five category labels:

| Category |
|---|
| Freezing |
| Cold |
| Moderate |
| Warm |
| Hot |

The training script derives classes from quantiles of the training target. These labels are presentation labels; they should not be interpreted as universally defined meteorological thresholds.

## Getting Started

### Prerequisites

- Python **3.14**
- pip
- The project training CSV, with the columns expected by `train_model.py`

Check your Python version:

```bash
python --version
```

The output should begin with `Python 3.14`.

### 1. Clone the repository

```bash
git clone https://github.com/PrahitViraajReddy/Global-Weather-Classifier.git
cd Global-Weather-Classifier
```

### 2. Create and activate a virtual environment (recommended)

Windows:

```bash
python -m venv .venv
.venv\\Scripts\\activate
```

macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Train the model

Place the training CSV in the project root using the filename and column structure expected by `train_model.py`, then run:

```bash
python train_model.py
```

This creates `weather_model.pkl`, which the app needs to make predictions.

### 5. Launch the app

```bash
streamlit run app.py
```

Streamlit will provide a local URL in the terminal. Open it in your browser.

## Project Structure

```text
Global-Weather-Classifier/
├── app.py              # Streamlit prediction interface
├── train_model.py      # Model training and serialization
├── requirements.txt    # Python dependencies
├── .gitignore          # Files excluded from version control
└── weather_model.pkl   # Generated model artifact (created during training)
```

## Model Inputs

The application uses the following features:

- Longitude and latitude
- Humidity and cloud cover
- Precipitation
- Wind speed and wind gust
- Visibility
- UV index
- Atmospheric pressure
- Ozone level

Inputs should use the units displayed in the app. Model predictions depend on the training data, feature definitions, and training process; they should not be treated as official meteorological advice.

## Limitations

- Predictions are only as reliable as the data and training procedure.
- The app does not retrieve current conditions or forecast future weather.
- Quantile-based categories depend on the temperature distribution in the training data.
- The class-probability chart represents model outputs, not a guarantee that a prediction is correct.
- The training script reserves a split of the data but currently fits the model on the training portion; evaluate on held-out data before using the model for serious decisions.

## Future Improvements

- **Integrate the latest live weather data** through a weather API, allowing users to retrieve current conditions for a selected location instead of entering every measurement manually.
- Explore combining live observations with the existing machine learning model while clearly distinguishing current-condition classification from future weather forecasting.
- Add reproducible evaluation metrics and a confusion matrix on held-out data.
- Document the dataset source, license, and preprocessing decisions.
- Add input validation and more robust handling of missing or invalid values.
- Track model versions and training configuration.
- Add a clearly documented deployment workflow that includes the trained model artifact.

## Author

**Prahit Viraaj Reddy**

- GitHub: [@PrahitViraajReddy](https://github.com/PrahitViraajReddy)
- LinkedIn: [Prahit Viraaj Reddy](https://linkedin.com/in/prahit-viraaj-reddy-madupu-5169332ba)

---

If you find this project useful, consider giving the repository a star.
