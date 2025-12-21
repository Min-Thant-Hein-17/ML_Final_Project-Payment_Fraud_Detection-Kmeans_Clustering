import pandas as pd
import numpy as np
import pickle
from flask import Flask, request, jsonify

app = Flask(__name__)

# --- 1) Load the trained artifact ---
try:
    with open('fraud_detection_model.pkl', 'rb') as f:
        artifact = pickle.load(f)
    
    model = artifact['model']
    EXPECTED_FEATURES = artifact['expected_features']
    print("✅ Model and schema loaded successfully.")
except FileNotFoundError:
    print("❌ Error: fraud_detection_model.pkl not found. Please run your training script first.")
    exit()

# --- 2) Helper function for feature engineering ---
def preprocess_input(data):
    """Transforms raw JSON input to match the training feature engineering."""
    df = pd.DataFrame([data])
    
    # Process Time (HH:MM:SS) to Continuous Hour
    if 'Transaction_Time' in df.columns:
        time_objs = pd.to_datetime(df['Transaction_Time'], format='%H:%M:%S')
        df['Time_Continuous'] = time_objs.dt.hour + time_objs.dt.minute / 60.0
    
    # Process Date to Day of Week (0-6)
    if 'Transaction_Date' in df.columns:
        df['Day_of_Week'] = pd.to_datetime(df['Transaction_Date']).dt.dayofweek
    
    # Ensure all columns exist, even if missing in JSON (imputer will handle NaNs)
    for col in EXPECTED_FEATURES:
        if col not in df.columns:
            df[col] = np.nan
            
    return df[EXPECTED_FEATURES]

# --- 3) Routes ---
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data
        input_json = request.get_json()
        if not input_json:
            return jsonify({"error": "No input data provided"}), 400

        # Preprocess and Align with EXPECTED_FEATURES
        processed_df = preprocess_input(input_json)

        # Predict cluster
        cluster_id = model.predict(processed_df)[0]

        # Note: Since this is K-Means, we return the cluster. 
        # You would map these cluster IDs to 'Fraud' or 'Legit' based on your analysis.
        return jsonify({
            "status": "success",
            "cluster_assigned": int(cluster_id),
            "message": f"Transaction assigned to cluster {cluster_id}"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "Model API is running"})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
