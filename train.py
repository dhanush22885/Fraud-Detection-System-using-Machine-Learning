# train.py (Updated)
import pandas as pd
import numpy as np
import joblib
import json
import os

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# --- Configuration ---
DATA_FILE = 'upi_transactions_unlabeled_10000.csv'
ARTIFACTS_DIR = 'artifacts'
PIPELINE_FILE = os.path.join(ARTIFACTS_DIR, 'iforest_pipeline.joblib')
CONFIG_FILE = os.path.join(ARTIFACTS_DIR, 'model_config.json')
MODEL_CONTAMINATION = 0.02 # Expected percentage of anomalies in the data

# --- Feature Engineering ---

def extract_upi_features(data, col_name):
    """Extracts features from a UPI ID string."""
    s = data[col_name].fillna('').astype(str)
    parts = s.str.split('@', expand=True)
    local = parts[0]
    handle = parts[1].fillna('')

    data[f'{col_name}_len'] = s.str.len()
    data[f'{col_name}_local_len'] = local.str.len()
    data[f'{col_name}_has_digits'] = local.str.match(r'.*\d.*').astype(int)
    data[f'{col_name}_handle'] = handle
    
    suspicious_keywords = ['help', 'refund', 'support', 'service', 'customer', 'care', 'urgent', 'verify', 'admin', 'official', 'alert']
    suspicious_pattern = '|'.join(suspicious_keywords)
    data[f'{col_name}_suspicious'] = s.str.lower().str.contains(suspicious_pattern, na=False).astype(int)
    
    return data

def process_data(data):
    """Applies all preprocessing and feature engineering steps."""
    # Check if 'timestamp' column exists before processing
    if 'timestamp' in data.columns:
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        data['hour'] = data['timestamp'].dt.hour
        data['day_of_week'] = data['timestamp'].dt.dayofweek
    else:
        # Create dummy time features if timestamp is not in the dataset
        # This is safer for datasets that might not have it
        data['hour'] = 0
        data['day_of_week'] = 0

    data = extract_upi_features(data, 'sender_upi')
    data = extract_upi_features(data, 'receiver_upi')
    
    return data

# --- Main Training Logic ---

def train_model():
    """Loads data, trains the model, and saves artifacts."""
    print("Starting model training process...")
 
    # Create artifacts directory if it doesn't exist
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    # 1. Load and Process Data
    try:
        print(f"Loading data from {DATA_FILE}...")
        data = pd.read_csv(DATA_FILE)
    except FileNotFoundError:
        print(f"Error: Data file '{DATA_FILE}' not found.")
        print("Please make sure the data file is in the correct location.")
        return

    data_processed = process_data(data.copy())
    
    # 2. Define Features
    # 'hour' and 'day_of_week' are now safely included
    numeric_features = [
        "amount", "vpa_age_days", "num_prev_txns_24h", "avg_amount_7d",
        "failed_pin_attempts", "location_lat", "location_long", "hour", "day_of_week",
        "sender_upi_len", "sender_upi_local_len", "sender_upi_has_digits", "sender_upi_suspicious",
        "receiver_upi_len", "receiver_upi_local_len", "receiver_upi_has_digits", "receiver_upi_suspicious",
        "is_recipient_new", "same_device_flag"
    ]
    
    # REMOVED 'sender_bank' and 'receiver_bank' as requested
    categorical_features = [
        "txn_type", "channel", "merchant_category",
        "sender_upi_handle", "receiver_upi_handle"
    ]

    # Ensure all defined features are actually in the processed DataFrame
    # This prevents errors if the source CSV changes
    all_features = numeric_features + categorical_features
    
    # Check for missing columns and handle them
    for col in all_features:
        if col not in data_processed.columns:
            print(f"Warning: Column '{col}' not found in data. Will be filled with 0 or 'unknown'.")
            if col in numeric_features:
                data_processed[col] = 0
            elif col in categorical_features:
                data_processed[col] = 'unknown'

    # Filter numeric_features and categorical_features to only those that exist
    numeric_features = [f for f in numeric_features if f in data_processed.columns]
    categorical_features = [f for f in categorical_features if f in data_processed.columns]

    # Fill NaNs for safety
    data_processed[numeric_features] = data_processed[numeric_features].fillna(0)
    data_processed[categorical_features] = data_processed[categorical_features].fillna('unknown')

    # 3. Create the Preprocessing Pipeline
    preprocessor = ColumnTransformer(transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features) # sparse_output=False for dense array
    ], remainder="drop")

    # 4. Define the Model
    model = IsolationForest(
        n_estimators=200, 
        contamination=MODEL_CONTAMINATION, 
        random_state=42,
        n_jobs=-1 # Use all available CPU cores
    )

    # 5. Create and Train the Full Pipeline
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("iforest", model)
    ])
    
    X = data_processed[numeric_features + categorical_features]
    print("Fitting the pipeline... This might take a moment.")
    pipeline.fit(X)
    print("Pipeline fitting complete.")

    # 6. Save the Pipeline
    joblib.dump(pipeline, PIPELINE_FILE)
    print(f"Model pipeline saved to: {PIPELINE_FILE}")

    # 7. Create and Save the Model Configuration
    # This config file is crucial for the Flask app
    # UPDATED: Removed bank info from dropdowns
    model_config = {
        'numeric_features': numeric_features,
        'categorical_features': categorical_features,
        'dropdown_options': {
            'txn_type': sorted(data_processed['txn_type'].unique().tolist()),
            'channel': sorted(data_processed['channel'].unique().tolist()),
            'merchant_category': sorted(data_processed['merchant_category'].unique().tolist()),
        }
    }

    with open(CONFIG_FILE, 'w') as f:
        json.dump(model_config, f, indent=4)
    print(f"Model configuration saved to: {CONFIG_FILE}")
    print("\nTraining process finished successfully!")

if __name__ == "__main__":
    train_model()