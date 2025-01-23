# -*- coding: utf-8 -*-  
"""  
SVM Training and Evaluation Script  
Created on Thu Nov 28 21:55:40 2024  

@author: Abc  
"""  

import pandas as pd  
from sklearn.model_selection import train_test_split, GridSearchCV  
from sklearn.svm import SVC  
from sklearn.preprocessing import StandardScaler, LabelEncoder  
from sklearn.metrics import accuracy_score, classification_report  
from sklearn.utils.class_weight import compute_class_weight  
import numpy as np  
import joblib  # To save the model

# Step 2: Train and Evaluate SVM Model  
def train_svm(csv_file):  
    # Load dataset  
    data = pd.read_csv(csv_file)  
    X = data.iloc[:, 2:].values  # Features  
    y = data["Label"].values  # Labels  
    
    # Encode labels  
    label_encoder = LabelEncoder()  
    y_encoded = label_encoder.fit_transform(y)  
    
    # Split dataset  
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)  
    
    # Scale features  
    scaler = StandardScaler()  
    X_train = scaler.fit_transform(X_train)  
    X_test = scaler.transform(X_test)  
    
    # Calculate class weights  
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)  
    class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}  
    
    # Perform Grid Search for Hyperparameter Tuning  
    param_grid = {  
        'C': [0.1, 1, 10, 100],  
        'kernel': ['linear', 'rbf', 'poly'],  
        'gamma': ['scale', 'auto']  
    }  
    grid = GridSearchCV(SVC(class_weight=class_weight_dict), param_grid, cv=5)  
    grid.fit(X_train, y_train)  
    
    # Best model  
    best_model = grid.best_estimator_  
    print(f"Best Parameters: {grid.best_params_}")  
    
    # Evaluate model  
    y_pred = best_model.predict(X_test)  
    accuracy = accuracy_score(y_test, y_pred)  
    print(f"\nAccuracy: {accuracy * 100:.2f}%")  
    print("\nClassification Report:")  
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))    

    # Save the trained model and the scaler
    joblib.dump(best_model, 'model/svm_model.pkl')  # Save the best trained model
    joblib.dump(scaler, 'model/scaler.pkl')         # Save the scaler used during training
    print("Model saved to 'model/svm_model.pkl'")
    print("Scaler saved to 'model/scaler.pkl'")

# Main Execution  
if __name__ == "__main__":  
    csv_file = "mantra_dataset_improved.csv"  # Path to the generated CSV  
    train_svm(csv_file)
