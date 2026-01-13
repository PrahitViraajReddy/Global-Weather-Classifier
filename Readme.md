

---

# 🌦️ **Global Weather Classification Using Machine Learning**

This project analyzes a global weather dataset and builds a machine learning model to classify temperature ranges based on various meteorological and air-quality features. It includes data exploration, preprocessing, model training, and performance evaluation using modern ML techniques.

---

## 📌 **Project Overview**

The goal of this project is to predict **temperature categories** (Freezing, Cold, Moderate, Warm, Hot) using multiple weather parameters such as:

* Humidity
* Cloud cover
* Precipitation
* Wind speed
* Visibility
* UV index
* Air quality data
* Latitude & Longitude

The model uses **RandomForestClassifier**, a powerful ensemble method, to learn patterns from a large dataset containing **116,933 global weather records**.

---

## 📂 **Key Features of the Project**

### **1. Data Loading & Exploration**

The script loads a large weather dataset and performs:

* Summary statistics (`describe()`)
* Missing value checks
* Duplicate checks
* Group-by analysis for trends (e.g., wind mph vs kph)
* Distribution exploration of major variables

This helps understand the structure and quality of the data.

---

### **2. Data Preprocessing**

The target variable `temperature_celsius` is transformed into **5 bins**:

* 0 → Freezing
* 1 → Cold
* 2 → Moderate
* 3 → Warm
* 4 → Hot

Feature columns include temperature-related, wind-related, and air-quality metrics.

---

### **3. Model Training**

The dataset is split using **train-test split (80/20)**.
A **RandomForestClassifier** is trained to classify the temperature category.

Key steps:

* Fitting the model
* Predicting on test data
* Balanced accuracy score
* Confusion matrix
* Classification report (precision/recall/F1)

---

### **4. ROC Curve & AUC Score**

The project includes **multi-class ROC analysis** using one-vs-rest technique.

It plots:

* ROC curve for each class
* AUC value for each class
* Combined graph visualizing classifier performance

It also computes **macro AUC**, which in this project reaches ~**0.96**, indicating strong model ranking performance.

---

## 📊 **Results Summary**

* **Balanced Accuracy:** ~0.71
* **Macro AUC:** ~0.96
* **Strong performance** for moderate → warm temperature predictions
* Model handles large global dataset efficiently

---

## 🛠️ **Technologies Used**

* Python
* Pandas, NumPy
* Scikit-learn
* Matplotlib
* Seaborn

---

## 🎯 **Purpose**

This project demonstrates:

* Applied machine learning on real-world environmental data
* Multi-class classification
* ROC/AUC evaluation
* Data analysis & model interpretation

It’s ideal for:

* Portfolio projects
* ML/AI learning
* Data analysis practice
* Weather analytics experiments

---

