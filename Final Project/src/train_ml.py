import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

def train_model():
    print("=" * 60)
    print("TRAINING LOAN APPROVAL PREDICTIVE MODEL (DATA SCIENCE)")
    print("=" * 60)
    
    if not os.path.exists("loan_approval_dataset.csv"):
        raise FileNotFoundError("Could not find 'loan_approval_dataset.csv' in the workspace.")
        
    print("Loading 'loan_approval_dataset.csv'...")
    df = pd.read_csv("loan_approval_dataset.csv")
    
    # Preprocess column headers
    df.columns = df.columns.str.strip()
    
    # Strip spaces from string columns
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].str.strip()
        
    # Map binary categoricals
    df['education'] = df['education'].map({'Graduate': 1, 'Not Graduate': 0})
    df['self_employed'] = df['self_employed'].map({'Yes': 1, 'No': 0})
    df['loan_status'] = df['loan_status'].map({'Approved': 1, 'Rejected': 0})
    
    # Define predictive features
    features = [
        'no_of_dependents', 'education', 'self_employed', 'income_annum',
        'loan_amount', 'loan_term', 'cibil_score', 'residential_assets_value',
        'commercial_assets_value', 'luxury_assets_value', 'bank_asset_value'
    ]
    X = df[features]
    y = df['loan_status']
    
    # Stratified Train-Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Fit Random Forest Classifier
    print("Training Random Forest Classifier on applicant profiles...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    model.fit(X_train, y_train)
    
    # Predict and evaluate
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    
    print(f"\nModel Performance Metrics:")
    print(f"  - Accuracy Score: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  - ROC-AUC Score : {roc_auc:.4f}")
    
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save model and importances
    model_package = {
        "model": model,
        "features": features,
        "metrics": {
            "accuracy": float(accuracy),
            "roc_auc": float(roc_auc),
            "report": report_dict
        },
        "feature_importances": dict(zip(features, model.feature_importances_.tolist()))
    }
    
    os.makedirs("src/models", exist_ok=True)
    pkl_path = "src/models/loan_classifier.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(model_package, f)
        
    print(f"\nModel saved successfully to '{pkl_path}'.")
    print("=" * 60)

if __name__ == "__main__":
    train_model()
