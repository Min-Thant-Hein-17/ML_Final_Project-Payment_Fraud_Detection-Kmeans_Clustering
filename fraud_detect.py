#using feature engineering stage!
import streamlit as st 
import pickle
import os
import pandas as pd
import numpy as np

#Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import silhouette_score


# 1. Load and Clean (Drop identifiers so select_dtypes works correctly)
# 1. Load and Clean
fd = pd.read_csv("dataset/luxury_cosmetics_fraud_analysis_2025.csv")

# 2. FEATURE ENGINEERING (Must match your app.py!)
# Convert Time string to a simple number
time_objs = pd.to_datetime(fd['Transaction_Time'], format='%H:%M:%S')
fd['Time_Continuous'] = time_objs.dt.hour + time_objs.dt.minute/60

# Convert Date to Day of Week
fd['Day_of_Week'] = pd.to_datetime(fd['Transaction_Date']).dt.dayofweek

# 3. DROP the messy columns BEFORE select_dtypes
cols_to_drop = ['Transaction_ID', 'Customer_ID', 'Fraud_Flag', 'IP_Address', 'Transaction_Date', 'Transaction_Time']
fd_cleaned = fd.drop(columns=[c for c in cols_to_drop if c in fd.columns])

# 4. NOW define your features
numerical_features = fd_cleaned.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = fd_cleaned.select_dtypes(include=['object']).columns.tolist()

# ... (The rest of your Pipeline and Pickle code remains the same) ...
# 3. Pipelines & ColumnTransformer (Your logic is good here)
cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

num_transformer = Pipeline(steps=[
    ('imputer', KNNImputer(n_neighbors=5)), # Mastery: Complex Imputation
    ('scaler', StandardScaler())           # Mastery: Scaling
])

preprocessor_instance = ColumnTransformer(transformers=[
    ('num', num_transformer, numerical_features),
    ('cat', cat_transformer, categorical_features)
])

# 4. The Final Model Pipeline (Expert Understanding)
model = Pipeline(steps=[
    ('preprocess', preprocessor_instance),
    ('cluster_model', KMeans(n_clusters=5, random_state=42, n_init='auto'))
])

# 5. Fit the WHOLE pipeline on the raw cleaned data
model.fit(fd_cleaned)

# 6. Save the ENTIRE pipeline (Preprocess + KMeans)
with open('fraud_detection_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# --- STREAMLIT SECTION ---
st.title("Fraud Detection in Luxury Cosmetics")

# Load the full pipeline
with open('fraud_detection_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    input_data = pd.read_csv(uploaded_file)
    
    # We must drop the same columns from the input data to avoid errors
    input_cleaned = input_data.drop(columns=[c for c in cols_to_drop if c in input_data.columns])
    
    # Since we saved the whole pipeline, we just call predict() on the raw data!
    # The pipeline handles the preprocessing automatically.
    predictions = loaded_model.predict(input_cleaned)
    
    input_data['Cluster'] = predictions
    st.write("Predictions (Cluster 0-4):")
    st.dataframe(input_data)


# 1. Ensure 'model' is the name of your final Pipeline variable
# 2. Open a new file in 'write-binary' (wb) mode
with open('fraud_detection_model.pkl', 'wb') as file:
    pickle.dump(model, file)


print("✅ Pickle file 'fraud_detection_model.pkl' has been created successfully!")
