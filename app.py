import os
import pickle
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# --- 1. Configuration & Setup ---
st.set_page_config(
    page_title="Luxury Fraud Detector",
    page_icon="🛡️",
    layout="wide"
)

# --- 2. Sidebar for Info ---
st.sidebar.title("🛡️ Fraud Sentinel")
logo_url = "https://talloiresnetwork.tufts.edu/wp-content/uploads//Parami-University-1.png"

if logo_url:
    # Fixed deprecated parameter based on your screenshot warning
    st.sidebar.image(logo_url, use_container_width=True) 
else:
    st.sidebar.subheader("Parami University")

st.sidebar.markdown("---")
st.sidebar.subheader("Owner's Information")
st.sidebar.write("**Name:** Min Thant Hein")
st.sidebar.write("**Student ID:** PIUS20230001")
st.sidebar.write("**Contact Email:** minthanthein@parami.edu.mm")
st.sidebar.markdown("---")

# --- 3. Load Model Pipeline ---
@st.cache_resource
def load_model(model_path: str):
    if not os.path.exists(model_path):
        st.error(f"Model file not found. Ensure '{model_path}' is in the folder.")
        st.stop()
    with open(model_path, "rb") as f:
        return pickle.load(f)

model = load_model("fraud_detection_model.pkl")

# --- 4. Main Application Layout ---
col_left, col_center, col_right = st.columns([1, 3, 1])
with col_center:
    st.title("🛡️ Luxury Cosmetics Fraud Detection")
    st.write("Enter transaction details below to identify which behavioral segment the activity belongs to.")

st.markdown("---")

# --- 5. Manual Input Form (Replacing File Uploader) ---
st.subheader("Transaction Details")

# Define columns for input fields
col1, col2, col3 = st.columns(3)
col4, col5, col6 = st.columns(3)

with col1:
    purchase_amount = st.number_input("Purchase Amount ($)", min_value=0.0, value=500.0, step=10.0)
with col2:
    customer_age = st.number_input("Customer Age", min_value=18, max_value=100, value=30)
with col3:
    footfall_count = st.number_input("Store Footfall Count", min_value=0, value=50)

with col4:
    loyalty_tier = st.selectbox("Loyalty Tier", options=['Gold', 'Silver', 'Bronze', 'None'])
with col5:
    payment_method = st.selectbox("Payment Method", options=['Credit Card', 'PayPal', 'Cash', 'Crypto'])
with col6:
    product_cat = st.selectbox("Product Category", options=['Skincare', 'Fragrance', 'Makeup', 'Sets'])

# Time and Day Engineering (matching your previous feature engineering)
col7, col8 = st.columns(2)
with col7:
    trans_hour = st.slider("Transaction Hour (0-23)", 0, 23, 14)
with col8:
    day_of_week = st.selectbox("Day of Week", options=[0, 1, 2, 3, 4, 5, 6], 
                               format_func=lambda x: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][x])

# --- 6. Prediction Logic ---
st.write("")
predict_btn = st.button("✨ Identify Transaction Cluster", type="primary")

if predict_btn:
    try:
        # Create a dictionary of features that matches your training exactly
        # Ensure 'Time_Continuous' matches the engineering you did in the notebook
        input_dict = {
            'Purchase_Amount': [float(purchase_amount)],
            'Customer_Age': [float(customer_age)],
            'Footfall_Count': [float(footfall_count)],
            'Customer_Loyalty_Tier': [loyalty_tier],
            'Payment_Method': [payment_method],
            'Product_Category': [product_cat],
            'Time_Continuous': [float(trans_hour)], 
            'Day_of_Week': [day_of_week]
        }
        
        input_df = pd.DataFrame(input_dict)

        # Step 2: Prediction via Pipeline
        cluster = model.predict(input_df)[0]

        # Display Result
        st.success(f"✅ Analysis Complete! This transaction belongs to **Cluster {cluster}**")
        
        # Mastery Point: Practical Interpretation Guide
        st.markdown("---")
        st.subheader("Cluster Interpretation Guide")
        interpretations = {
            0: "Standard High-Value Purchase",
            1: "Potential Anomaly (High Frequency/Low Value)",
            2: "Typical Retail Customer",
            3: "New Account / Rare Transaction Pattern",
            4: "Verified VIP Segment"
        }
        st.info(f"**Cluster {cluster} Insight:** {interpretations.get(cluster, 'Unclassified Segment')}")

    except Exception as e:
        st.error(f"Error processing inputs: {e}")
