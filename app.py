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

# --- 2. Sidebar for Info (Car Industry style) ---
st.sidebar.title("🛡️ Fraud Sentinel")
logo_url = "https://talloiresnetwork.tufts.edu/wp-content/uploads//Parami-University-1.png"

if logo_url:
    st.sidebar.image(logo_url, use_column_width=True)
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
        st.error("Model file not found. Ensure 'fraud_detection_model.pkl' is in the folder.")
        st.stop()
    with open(model_path, "rb") as f:
        return pickle.load(f)

model = load_model("fraud_detection_model.pkl")

# --- 4. Main Application Layout ---
col_left, col_center, col_right = st.columns([1, 3, 1])
with col_center:
    st.title("🛡️ Luxury Cosmetics Fraud Detection")
    st.write("Upload transaction data to group customers and identify suspicious behavior clusters.")

st.markdown("---")

# File Uploader
uploaded_file = st.file_uploader("📂 Upload Transaction CSV File", type="csv")

if uploaded_file is not None:
    input_data = pd.read_csv(uploaded_file)
    
    # Simple Progress Bar
    with st.spinner('Analyzing patterns...'):
        # Step 1: Pre-process (Drop same columns as training)
        cols_to_drop = ['Transaction_ID', 'Customer_ID', 'Fraud_Flag', 'IP_Address', 'Transaction_Date', 'Transaction_Time']
        input_cleaned = input_data.drop(columns=[c for c in cols_to_drop if c in input_data.columns])
        
        # Step 2: Prediction
        # The pipeline handles KNN Imputation and Scaling automatically!
        predictions = model.predict(input_cleaned)
        input_data['Cluster'] = predictions

    # --- 5. Display Results in Columns ---
    st.success("✅ Analysis Complete!")
    
    tab1, tab2 = st.tabs(["📄 Data View", "📊 Cluster Visualization"])
    
    with tab1:
        st.subheader("Transaction List with Assigned Clusters")
        st.dataframe(input_data, use_container_width=True)
        
    with tab2:
        st.subheader("Cluster Separation (PCA Map)")
        # Required for 'Mastery' in Visualization (3 pts)
        X_preprocessed = model.named_steps['preprocess'].transform(input_cleaned)
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_preprocessed)
        
        pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
        pca_df['Cluster'] = predictions
        
        fig, ax = plt.subplots()
        sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='Cluster', palette='viridis', ax=ax)
        plt.title("2D Visual of Hidden Patterns")
        st.pyplot(fig)
        
        st.info("Clusters with very high Purchase Amounts or unusual profiles often represent fraud risk.")

else:
    st.info("Please upload a CSV file to begin the clustering analysis.")
