# app.py (Updated with Hybrid Logic)
from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import json
import os

app = Flask(__name__)

# --- Load Model and Configuration at Startup ---
ARTIFACTS_DIR = 'artifacts'
PIPELINE_FILE = os.path.join(ARTIFACTS_DIR, 'iforest_pipeline.joblib')
CONFIG_FILE = os.path.join(ARTIFACTS_DIR, 'model_config.json')

try:
    pipeline = joblib.load(PIPELINE_FILE)
    with open(CONFIG_FILE, 'r') as f:
        model_config = json.load(f)
    print("Model pipeline and configuration loaded successfully.")
except FileNotFoundError:
    print("Error: Model artifacts not found. Please run train.py first.")
    pipeline = None
    model_config = None

# --- Feature Engineering for Single Prediction ---

def extract_upi_features_single(upi_string):
    """Extracts features from a single UPI ID string."""
    s = str(upi_string or "").strip()
    s_lower = s.lower()
    parts = s.split('@')
    local = parts[0] if parts else ""
    handle = parts[1] if len(parts) > 1 else ""

    suspicious_keywords = ['help', 'refund', 'support', 'service', 'customer', 'care', 'urgent', 'verify', 'admin', 'official', 'alert','lottery','win','claim','prize','payback','offer']

    features = {
        'len': len(s),
        'local_len': len(local),
        'has_digits': int(any(char.isdigit() for char in local)),
        'handle': handle if handle != "" else 'unknown',
        'suspicious': int(any(k in s_lower for k in suspicious_keywords))
    }
    return features

# --- Helper Functions for Data Cleaning ---

def to_float(x, default=0.0):
    try:
        return float(x)
    except:
        return default

def to_int(x, default=0):
    try:
        return int(float(x))
    except:
        return default

# --- Flask Routes ---

@app.route('/', methods=['GET'])
def index():
    """Renders the main input form."""
    if not model_config:
        return "Model not loaded. Please check server logs.", 500

    # Pass dropdown options to the template
    # This will now only pass 'txn_type', 'channel', etc. (no banks)
    return render_template('index.html', options=model_config['dropdown_options'])

@app.route('/predict', methods=['POST'])
def predict():
    """Receives form data, processes it, and returns a prediction."""
    if not pipeline or not model_config:
        return render_template('result.html', error="Model is not available. Please run training first.")

    try:
        # 1. Collect data from the form
        form_data = request.form.to_dict()

        # 2. Extract key features for rule-based checks
        amount = to_float(form_data.get('amount', 0))
        sender_upi_features = extract_upi_features_single(form_data.get('sender_upi', ''))
        receiver_upi_features = extract_upi_features_single(form_data.get('receiver_upi', ''))

        # --- NEW: 3. HARD RULE ENGINE (Rule 2 & 7) ---
        # Checks for definite fraud before running the model.
        hard_fraud_reasons = []
        
        # Rule 2: Amount exceeds 1 Lakh
        if amount > 100000:
            hard_fraud_reasons.append(f"Amount ({amount:,.2f}) exceeds 1 Lakh limit.")
            
        # Rule 7: Suspicious keywords in UPI
        if sender_upi_features['suspicious']:
            hard_fraud_reasons.append("Sender UPI contains suspicious keywords (e.g., 'support', 'refund').")
        
        if receiver_upi_features['suspicious']:
            hard_fraud_reasons.append("Receiver UPI contains suspicious keywords (e.g., 'support', 'refund').")

        # If any hard rule is met, flag as fraud immediately
        if hard_fraud_reasons:
            result = {
                'is_fraud': True,
                'status': 'FLAGGED AS FRAUDULENT (Hard Rule)',
                'score': "-1.0000 (Rule Violation)",
                'advice': "This transaction was blocked due to high-risk rules: " + " ".join(hard_fraud_reasons)
            }
            return render_template('result.html', result=result)

        # --- NEW: 4. HEURISTIC SCORE ENGINE (Rules 1, 3, 4, 5) ---
        # Calculates a risk score for "borderline" cases.
        heuristic_score = 0
        heuristic_reasons = []
        
        # Rule 1: 'request' mode increases risk
        if form_data.get('txn_type') == 'request':
            heuristic_score += 1
            heuristic_reasons.append("Payment Request")

        # Rule 3: More than 5 failed PIN attempts
        failed_pins = to_int(form_data.get('failed_pin_attempts', 0))
        if failed_pins > 5:
            heuristic_score += 2  # Higher weight for this
            heuristic_reasons.append(f"{failed_pins} Failed PINs")

        # Rule 4: Sender VPA age < 10 days
        vpa_age = to_int(form_data.get('vpa_age_days', 99))
        if vpa_age < 10:
            heuristic_score += 1
            heuristic_reasons.append("New Sender VPA (<10 days)")
            
        # Rule 5: Recipient is new
        if to_int(form_data.get('is_recipient_new', 0)) == 1:
            heuristic_score += 1
            heuristic_reasons.append("New Recipient")

        # --- 5. Prepare data for ML Model ---
        data = {}
        
        # Add numeric features
        data['amount'] = amount
        data['vpa_age_days'] = vpa_age
        data['num_prev_txns_24h'] = to_int(form_data.get('num_prev_txns_24h', 0))
        data['avg_amount_7d'] = to_float(form_data.get('avg_amount_7d', 0))
        data['failed_pin_attempts'] = failed_pins
        data['is_recipient_new'] = to_int(form_data.get('is_recipient_new', 0))
        data['same_device_flag'] = to_int(form_data.get('same_device_flag', 0))
        data['location_lat'] = to_float(form_data.get('location_lat', 0.0))
        data['location_long'] = to_float(form_data.get('location_long', 0.0))

        # Add time features
        now = datetime.now()
        data['hour'] = now.hour
        data['day_of_week'] = now.weekday()

        # Add categorical features (now skips banks automatically)
        for feature, options in model_config['dropdown_options'].items():
            val = form_data.get(feature)
            if val is None or val == '':
                val = options[0] if isinstance(options, list) and len(options) > 0 else 'unknown'
            data[feature] = val

        # Add UPI-derived features
        data['sender_upi_len'] = sender_upi_features['len']
        data['sender_upi_local_len'] = sender_upi_features['local_len']
        data['sender_upi_has_digits'] = sender_upi_features['has_digits']
        data['sender_upi_suspicious'] = sender_upi_features['suspicious']
        data['sender_upi_handle'] = sender_upi_features['handle']

        data['receiver_upi_len'] = receiver_upi_features['len']
        data['receiver_upi_local_len'] = receiver_upi_features['local_len']
        data['receiver_upi_has_digits'] = receiver_upi_features['has_digits']
        data['receiver_upi_suspicious'] = receiver_upi_features['suspicious']
        data['receiver_upi_handle'] = receiver_upi_features['handle']

        # 6. Create DataFrame and ensure correct order
        sample_df = pd.DataFrame([data])
        
        numeric_features = model_config.get('numeric_features', [])
        categorical_features = model_config.get('categorical_features', [])

        for f in numeric_features:
            if f not in sample_df.columns: sample_df[f] = 0
        for f in categorical_features:
            if f not in sample_df.columns: sample_df[f] = 'unknown'

        ordered_features = numeric_features + categorical_features
        sample_df = sample_df[ordered_features]

        # 7. Make ML Prediction
        prediction = pipeline.predict(sample_df)[0] # -1 anomaly, 1 normal
        transformed = pipeline.named_steps['preprocessor'].transform(sample_df)
        score = pipeline.named_steps['iforest'].decision_function(transformed)[0]

        model_is_anomaly = (prediction == -1)
        
        # --- NEW: 8. FINAL HYBRID DECISION ---
        is_fraud = False
        status = ""
        advice = ""
        
        HEURISTIC_THRESHOLD = 3 # Triggers fraud if score is 3 or more

        if model_is_anomaly:
            is_fraud = True
            status = "FLAGGED AS POTENTIALLY FRAUDULENT (ML Anomaly)"
            advice = "This transaction is highly anomalous and was flagged by our security model."
            if heuristic_reasons:
                advice += " Contributing factors: " + ", ".join(heuristic_reasons)
        
        elif heuristic_score >= HEURISTIC_THRESHOLD:
            is_fraud = True
            status = "FLAGGED AS POTENTIALLY FRAUDULENT (Risk Rules)"
            advice = f"This transaction was flagged due to a combination of risk factors: {', '.join(heuristic_reasons)}."
        
        else:
            is_fraud = False
            status = "SEEMS LEGITIMATE"
            advice = "This transaction appears to be normal."
            if heuristic_score > 0:
                advice += f" (Note: Low risk factors detected: {', '.join(heuristic_reasons)})"

        result = {
            'is_fraud': is_fraud,
            'status': status,
            'score': f"{score:.4f} (ML Score) | {heuristic_score} (Risk Score)",
            'advice': advice
        }

        return render_template('result.html', result=result)

    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        return render_template('result.html', error=f"An error occurred: {e}")


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)