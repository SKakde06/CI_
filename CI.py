import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    mean_squared_error, r2_score, 
    accuracy_score, precision_score, 
    recall_score, f1_score, 
    classification_report, 
    silhouette_score
)
from sklearn.decomposition import PCA
import logging

# Load the dataset
data = pd.read_csv('Customer Purchasing Behaviors.csv')

# Data Preprocessing
print("Dataset Information:")
print(data.info())
print("\nSummary Statistics:")
print(data.describe())

# Check for missing values
print("\nMissing Values:")
print(data.isnull().sum())

# Regression Model
X = data.drop(columns=['user_id', 'purchase_amount', 'region'])
y = data['purchase_amount']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train multiple regression models
regression_models = {
    "Linear Regression": LinearRegression(),
    "Random Forest Regressor": RandomForestRegressor(random_state=42)
}

print("\n--- Regression Model Results ---")
regression_results = {}
for name, model in regression_models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)
    
    regression_results[name] = {'RMSE': rmse, 'R^2': r2}
    
    print(f"{name}:")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  R^2 Score: {r2:.2f}\n")

# Classification Model
X_class = data.drop(columns=['user_id', 'region'])
y_class = data['region']
X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X_class, y_class, test_size=0.2, random_state=42)

classification_models = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "Random Forest Classifier": RandomForestClassifier(random_state=42)
}

print("\n--- Classification Model Results ---")
for name, model in classification_models.items():
    model.fit(X_train_class, y_train_class)
    y_pred_class = model.predict(X_test_class)
    
    print(f"{name}:")
    print(f"  Accuracy: {accuracy_score(y_test_class, y_pred_class):.2f}")
    print(classification_report(y_test_class, y_pred_class, zero_division=0))

# Clustering Models
X_cluster = data.drop(columns=['user_id', 'region'])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

clustering_models = {
    "KMeans": KMeans(n_clusters=3, random_state=42),
    "Agglomerative Clustering": AgglomerativeClustering(n_clusters=3),
    "Gaussian Mixture": GaussianMixture(n_components=3, random_state=42)
}

print("\n--- Clustering Model Results ---")
clustering_results = {}
for name, model in clustering_models.items():
    if name == "Gaussian Mixture":
        clusters = model.fit_predict(X_scaled)
    else:
        clusters = model.fit_predict(X_scaled)
    
    sil_score = silhouette_score(X_scaled, clusters)
    clustering_results[name] = sil_score
    
    print(f"{name} Silhouette Score: {sil_score:.2f}")

# Hyperparameter Tuning for Clustering
print("\n--- Clustering Hyperparameter Tuning ---")
for name in ["KMeans", "Agglomerative Clustering", "Gaussian Mixture"]:
    best_silhouette = -1
    best_params = {}
    
    for n_clusters in range(2, 6):
        if name == "KMeans":
            clusterer = KMeans(n_clusters=n_clusters, random_state=42)
        elif name == "Agglomerative Clustering":
            clusterer = AgglomerativeClustering(n_clusters=n_clusters)
        else:
            clusterer = GaussianMixture(n_components=n_clusters, random_state=42)
        
        if name == "Gaussian Mixture":
            clusters = clusterer.fit_predict(X_scaled)
        else:
            clusters = clusterer.fit_predict(X_scaled)
        
        sil_score = silhouette_score(X_scaled, clusters)
        
        if sil_score > best_silhouette:
            best_silhouette = sil_score
            best_params = {"n_clusters" if name != "Gaussian Mixture" else "n_components": n_clusters}
    
    print(f"{name} Best Parameters: {best_params}")
    print(f"Best Silhouette Score: {best_silhouette:.2f}")

# Final Model Performance Logging
logging.basicConfig(filename='model_performance.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

logging.info("Regression Model Results: " + str(regression_results))
logging.info("Clustering Model Silhouette Scores: " + str(clustering_results))

print("\nModel performance logged to 'model_performance.log'")
