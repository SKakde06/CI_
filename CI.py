import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def load_and_preprocess_data(file_path):
    """Load and preprocess the data in a single pass"""
    data = pd.read_csv(file_path)
    
    # Create derived features in one go
    data['high_spender'] = (data['purchase_amount'] > 300).astype(int)
    
    # Encode categorical features
    label_encoder = LabelEncoder()
    data['region'] = label_encoder.fit_transform(data['region'])
    
    return data

def create_visualization_summary(data):
    """Create essential visualizations in a single figure"""
    fig = plt.figure(figsize=(15, 10))
    
    # Plot 1: Purchase Amount Distribution
    plt.subplot(2, 2, 1)
    sns.histplot(data['purchase_amount'], kde=True)
    plt.title('Purchase Amount Distribution')
    
    # Plot 2: Income vs Purchase
    plt.subplot(2, 2, 2)
    sns.scatterplot(data=data, x='annual_income', y='purchase_amount')
    plt.title('Income vs Purchase')
    
    # Plot 3: Region Distribution
    plt.subplot(2, 2, 3)
    sns.countplot(data=data, x='region')
    plt.title('Region Distribution')
    
    # Plot 4: High Spender Distribution
    plt.subplot(2, 2, 4)
    sns.countplot(data=data, x='high_spender')
    plt.title('High Spender Distribution')
    
    plt.tight_layout()
    plt.show()

def train_and_evaluate_models(data):
    """Train and evaluate models in an optimized way"""
    # Prepare features and target
    X = data.drop(columns=['user_id', 'purchase_amount', 'high_spender'])
    y = data['high_spender']
    
    # Single scaling operation
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data once
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # Train Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    
    # Train KMeans (optimal k=3 based on previous analysis)
    kmeans = KMeans(n_clusters=3, random_state=42)
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    # Calculate all metrics at once
    metrics = {
        'accuracy': accuracy_score(y_test, rf_pred),
        'precision': precision_score(y_test, rf_pred),
        'recall': recall_score(y_test, rf_pred),
        'f1': f1_score(y_test, rf_pred),
        'silhouette': silhouette_score(X_scaled, cluster_labels)
    }
    
    return metrics, cluster_labels

def main():
    # Load and process data
    data = load_and_preprocess_data('/content/Customer Purchasing Behaviors.csv')
    
    # Create visualizations
    create_visualization_summary(data)
    
    # Train and evaluate models
    metrics, clusters = train_and_evaluate_models(data)
    
    # Print results
    print("\nModel Performance Metrics:")
    for metric, value in metrics.items():
        print(f"{metric.capitalize()}: {value:.3f}")

if __name__ == "__main__":
    main()
