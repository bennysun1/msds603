import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import yaml

# Load parameters if using params.yaml
try:
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)["preprocessing"]
except:
    params = {
        "test_size": 0.2,
        "random_state": 42
    }

def load_data(filepath):
    """Load the heart disease dataset"""
    return pd.read_parquet(filepath)

def preprocess_data(df):
    """Preprocess the heart disease data"""
    # Create a copy to avoid modifying the original data
    df = df.copy()
    
    # Handle any missing values if they exist
    df = df.dropna()
    
    # Separate features and target
    X = df.drop("HeartDisease", axis=1)
    y = df["HeartDisease"]
    
    # Identify categorical and numerical columns
    categorical_cols = X.select_dtypes(include=["object"]).columns
    numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns
    
    # Convert categorical variables
    X = pd.get_dummies(X, columns=categorical_cols)
    
    # Scale numerical features
    scaler = StandardScaler()
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
    
    return X, y

def split_and_save_data(X, y):
    """Split the data and save train/test sets"""
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=params["test_size"],
        random_state=params["random_state"],
        stratify=y
    )
    
    # Create DataFrames for saving
    train_data = pd.concat([X_train, y_train], axis=1)
    test_data = pd.concat([X_test, y_test], axis=1)
    
    # Save processed datasets
    train_data.to_parquet("data/processed_train.parquet")
    test_data.to_parquet("data/processed_test.parquet")

if __name__ == "__main__":
    # Load raw data
    raw_data = load_data("data/heart_df.parquet")
    
    # Preprocess
    X_processed, y = preprocess_data(raw_data)
    
    # Split and save
    split_and_save_data(X_processed, y)