import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import yaml
import os

# Column names for the adult dataset
COLUMNS = [
    'age', 'workclass', 'fnlwgt', 'education', 'education-num',
    'marital-status', 'occupation', 'relationship', 'race', 'sex',
    'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'
]

# Load parameters from params.yaml
try:
    with open("params.yaml", 'r') as f:
        params = yaml.safe_load(f)['preprocessing']
except:
    params = {
        'test_size': 0.2,
        'random_state': 42,
        'input_path': 'data/adult.data',
        'train_output_path': 'data/processed_train.csv',
        'test_output_path': 'data/processed_test.csv'
    }

def load_data(filepath):
    """Load the raw dataset"""
    # Read CSV without headers and assign column names
    df = pd.read_csv(filepath, header=None, names=COLUMNS)
    # Clean whitespace from string columns
    object_columns = df.select_dtypes(include=['object']).columns
    df[object_columns] = df[object_columns].apply(lambda x: x.str.strip())
    return df

def preprocess_data(df):
    """Preprocess the data"""
    # Handle missing values (represented as '?')
    df = df.replace('?', np.nan)
    df = df.dropna()
    
    # Convert categorical variables
    categorical_columns = df.select_dtypes(include=['object']).columns
    df = pd.get_dummies(df, columns=list(set(categorical_columns) - {'income'}))
    
    # Scale numerical features
    scaler = StandardScaler()
    numerical_columns = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
    
    return df

def split_and_save_data(df):
    """Split the data and save train/test sets"""
    # Clean target variable
    df['income'] = df['income'].map({'>50K': 1, '<=50K': 0})
    
    X = df.drop('income', axis=1)
    y = df['income']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=params['test_size'],
        random_state=params['random_state']
    )
    
    # Save processed datasets
    train_data = pd.concat([X_train, y_train], axis=1)
    test_data = pd.concat([X_test, y_test], axis=1)
    
    # Create data directory if it doesn't exist
    os.makedirs(os.path.dirname(params['train_output_path']), exist_ok=True)
    
    train_data.to_csv(params['train_output_path'], index=False)
    test_data.to_csv(params['test_output_path'], index=False)

if __name__ == "__main__":
    # Load raw data
    raw_data = load_data(params['input_path'])
    
    # Preprocess
    processed_data = preprocess_data(raw_data)
    
    # Split and save
    split_and_save_data(processed_data)