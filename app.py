import os
import pickle
import pandas as pd
import streamlit as st

# --- 1. Basic Setup & Owner Info ---
st.set_page_config(page_title="Fraud Detector", layout="wide")

st.sidebar.title("🛡️ Fraud Sentinel")
st.sidebar.image("https://talloiresnetwork.tufts.edu/wp-content/uploads//Parami-University-1.png", use_container_width=True)
st.sidebar.markdown("---")
st.sidebar.subheader("Owner's Information")
st.sidebar.write("**Name:** Min Thant Hein")
st.sidebar.write("**ID:** PIUS20230001")

# --- 2. Load the Model (The 'Brain' of our App) ---
@st.cache_resource
def load_model():
    with open("fraud_detection_model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# --- 3. User Interface (The 'Front-End') ---
st.title("🛡️ Luxury Cosmetics Fraud Detection")
st.write("Enter details to identify the behavioral cluster.")

# Create input fields in a clean grid
col1, col2, col3 = st.columns(3)
with col1:
    amt = st.number_input("Purchase Amount ($)", value=500.0)
    loyalty = st.selectbox("Loyalty Tier", ["Gold", "Silver", "Bronze", "None"])
with col2:
    age = st.number_input("Customer Age", value=30)
    pay = st.selectbox("Payment Method", ["Credit Card", "PayPal", "Cash", "Crypto"])
with col3:
    foot = st.number_input("Store Footfall", value=50)
    cat = st.selectbox("Product Category", ["Skincare", "Fragrance", "Makeup"])

# Time engineering sliders
hour = st.slider("Transaction Hour", 0, 23, 14)
day = st.selectbox("Day of Week", [0, 1, 2, 3, 4, 5, 6], format_func=lambda x: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][x])

# --- 4. Prediction Logic (The 'Back-End') ---
if st.button("✨ Identify Transaction Cluster", type="primary"):
    # Step 1: Wrap inputs into a DataFrame (must match training columns exactly!)
    input_df = pd.DataFrame({
        'Purchase_Amount': [float(amt)],
        'Customer_Age': [float(age)],
        'Footfall_Count': [float(foot)],
        'Time_Continuous': [float(hour)],
        'Day_of_Week': [day],
        'Customer_Loyalty_Tier': [loyalty],
        'Payment_Method': [pay],
        'Product_Category': [cat]
    })

    # Step 2: Re-order to match training EXACTLY
    # All these lines MUST be indented 1 tab (4 spaces) inside the 'if' block
    final_features = [
        'Purchase_Amount', 'Customer_Age', 'Footfall_Count', 'Time_Continuous', 
        'Day_of_Week', 'Customer_Loyalty_Tier', 'Payment_Method', 'Product_Category'
    ]
    input_df = input_df[final_features]

    # Step 3: Use the model to predict
    result = model.predict(input_df)[0]

    # Step 4: Show the output clearly
    st.markdown("---")
    st.success(f"### ✅ Transaction Identified: Cluster {result}")
    
    # Practical Usefulness: Explain what the cluster means
    if result == 0:
        st.info("💡 **Insight:** This represents a standard high-value customer.")
    else:
        st.warning("⚠️ **Insight:** This behavior matches patterns often flagged for review.")
