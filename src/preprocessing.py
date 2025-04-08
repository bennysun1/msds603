import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import yaml

# Load parameters
try:
    with open("params.yaml", 'r') as f:
        params = yaml.safe_load(f)['preprocessing']
except:
    params = {
        'test_size': 0.2,
        'val_size': 0.2,
        'random_state': 42
    }

def create_preprocessing_pipeline(categorical_cols, numerical_cols):
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])
    
    return preprocessor

if __name__ == "__main__":
    # Load data
    heart_df = pd.read_parquet('data/heart_df.parquet')
    
    # Split features and target
    X = heart_df.drop('HeartDisease', axis=1)
    y = heart_df['HeartDisease']
    
    # Get column types
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Create train/val/test splits
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, 
        test_size=params['test_size'], 
        random_state=params['random_state']
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, 
        test_size=params['val_size'], 
        random_state=params['random_state']
    )
    
    # Save splits
    X_train.to_parquet('data/X_train.parquet')
    X_val.to_parquet('data/X_val.parquet')
    X_test.to_parquet('data/X_test.parquet')
    pd.Series(y_train).to_csv('data/y_train.csv')
    pd.Series(y_val).to_csv('data/y_val.csv')
    pd.Series(y_test).to_csv('data/y_test.csv')