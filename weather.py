import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score,classification_report,confusion_matrix,roc_auc_score,roc_curve,auc
from sklearn.preprocessing import label_binarize
import joblib
import streamlit as st
try:
    df=pd.read_csv("GlobalWeatherRepository.csv")
except FileNotFoundError:
    print("File Not Found ")
    exit(0)
#Data handling and preprocessing
print(df.head())
print(df.info())
print(df.describe())
print(df.dropna())
print(df.isnull().sum())
print(df.duplicated())
print(df.groupby('latitude')['longitude'].mean())
print(df.groupby('wind_mph')['wind_kph'].mean())
print(df.groupby('feels_like_celsius')['feels_like_fahrenheit'].mean())
print(df.groupby('visibility_km')['visibility_miles'].mean())
#Linear Regression
X=df[['longitude','latitude','humidity','cloud','precip_in','wind_kph','visibility_km','uv_index','gust_mph','pressure_mb','air_quality_Ozone']]
y=pd.qcut(df['temperature_celsius'], q=5,labels=False,duplicates="drop")
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
model=RandomForestClassifier()
model.fit(X_train,y_train)
joblib.dump(model, "weather_model.pkl")
print("Model saved as weather_model.pkl")

predicted=model.predict(X_test)
print(predicted)
#checking the model Accuracy
accuracy=balanced_accuracy_score(y_test,predicted)
print(accuracy)
confusionmatrix=confusion_matrix(y_test,predicted)
print(confusionmatrix)
#classification report
print(classification_report(y_test,predicted))
#roc curve and score

predict=model.predict_proba(X_test)
y_test_bin = label_binarize(y_test, classes=[0, 1, 2, 3, 4])
n_classes = 5


colors = ['blue', 'green', 'red', 'orange', 'purple']
bin_names = ['Freezing', 'Cold', 'Moderate', 'Warm', 'Hot']
n_classes=5
for i in range(n_classes):
    # Calculate ROC curve for each class
    fpr, tpr, thresholds = roc_curve(y_test_bin[:, i], predict[:, i])
    roc_auc_class = auc(fpr, tpr)
    plt.plot(fpr, tpr, color=colors[i], lw=2, label=f'{bin_names[i]} (Bin {i}) - AUC = {roc_auc_class:.3f}')
plt.show()

    
auc=roc_auc_score(y_test,predict,average='macro',max_fpr=None,labels=None,multi_class='ovo')
print(auc)
