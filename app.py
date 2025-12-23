# app.py
import pickle
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Fraud Sentinel", layout="wide")

st.sidebar.title("🛡️ Project X")
st.sidebar.image("https://talloiresnetwork.tufts.edu/wp-content/uploads//Parami-University-1.png", use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.subheader("Owner's Information")
st.sidebar.write("**Name:** Min Thant Hein")
st.sidebar.write("**ID:** PIUS20230001")
st.sidebar.write(f"**Contact Email:** minthanthein@parami.edu.mm ")
st.sidebar.write("**Course:** Introduction to Machine Learning")
st.sidebar.write("**Professor:** Dr. Nwe Nwe Htay Win")
st.sidebar.markdown("---")

# Adding Reliable Links in the Sidebar
with st.expander("🔗 Reliable Websites and Resources"):
    st.write("🔗 Reliable Websites and Resources")
# Using st.sidebar.markdown to include the link
    st.write("Check **Cybersecurity Best Practices** at [Cisa.gov](https://www.cisa.gov/topics/cybersecurity-best-practices).")
    st.write("**Latest Industry News** from [University of Sandiego](https://onlinedegrees.sandiego.edu/top-cyber-security-blogs-websites/).")

    st.write("---")


with st.expander("ℹ️ User Guide: Dos and Don'ts for Fraud Detection"):
    col_do, col_dont = st.columns(2)
    with col_do:
        st.success("### ✅ Dos")
        st.write("- **Check High-Value Clusters:** Prioritize manual reviews for Cluster 1.")
        st.write("- **Verify Night Transactions:** Use the slider to check risk at late hours.")
        st.write("- **Validate VIPs:** Reward loyal customers in Cluster 3 with fast-track shipping.")
    with col_dont:
        st.error("### ❌ Don'ts")
        st.write("- **Ignore Age Alerts:** Never approve transactions with impossible ages.")
        st.write("- **Blindly Trust 0:** Even 'Standard' clusters need periodic audit.")
        st.write("- **Input Raw Data:** Ensure currency is converted to USD before entry.")

st.markdown("---")


@st.cache_resource
def load_model():
    with open("fraud_detection_Final_model.pkl", "rb") as f:
        return pickle.load(f)

artifact = load_model()
model = artifact['model']
EXPECTED_FEATURES = artifact['expected_features']

st.title("🛡️ Luxury Cosmetics & Payment Transaction Fraud Detection")
st.write("Professional Decision Support System for Loss Prevention Teams.")
st.markdown("---")


col1, col2, col3 = st.columns(3)
with col1:
    amt = st.number_input("Purchase Amount ($)", value=100.0)
    loyalty = st.selectbox("Loyalty Tier", ["Gold", "Silver", "Bronze", "None"])
with col2:
    age = st.number_input("Customer Age", value=25)
    pay = st.selectbox("Payment Method", ["Credit Card", "PayPal", "Cash", "Crypto"])
with col3:
    foot = st.number_input("Store Footfall", value=50)
    cat = st.selectbox("Product Category", ["Skincare", "Fragrance", "Makeup"])

hour = st.slider("Transaction Hour", 0, 23, 14)
day = st.selectbox("Day of Week", list(range(7)), format_func=lambda x: ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][x])

if st.button("✨ Identify Cluster", type="primary"):
    # 1. Using Human Logic Filter 
    if age > 95 or age < 16:
        st.error("🚨 ALERT: Physically impossible age detected. Manual fraud investigation required.")
    else:
        input_df = pd.DataFrame([{
            'Purchase_Amount': float(amt), 'Customer_Age': float(age), 'Footfall_Count': float(foot),
            'Time_Continuous': float(hour), 'Day_of_Week': int(day),
            'Customer_Loyalty_Tier': loyalty, 'Payment_Method': pay, 'Product_Category': cat
        }], columns=EXPECTED_FEATURES)

        cluster = int(model.predict(input_df)[0])
        
        # 2. Persona Mapping whether it is normal, fraud or suspicious!
        cluster_map = {
            0: {"name": "Standard Buyer", "color": "blue", "rec": "✅ **Action: Auto-Approve.** This profile fits typical shopping patterns. No further action needed.", "risk": "Low"},
            1: {"name": "High-Value Anomaly", "color": "red", "rec": "🚨 **Action: IMMEDIATE HOLD.** Large amount outlier. Contact customer via phone to verify identity before shipping.", "risk": "Extremely High"},
            2: {"name": "Night-time Suspicious", "color": "orange", "rec": "⚠️ **Action: Manual Review.** Transaction occurred at an unusual hour. Cross-reference IP address with shipping address.", "risk": "High"},
            3: {"name": "Verified VIP", "color": "green", "rec": "💎 **Action: Priority Handling.** Trusted loyalty member. Approve and apply complimentary express shipping.", "risk": "Safe"}
        }
        
        res = cluster_map[cluster]
        st.markdown(f"### Result: :{res['color']}[{res['name']} (Cluster {cluster})]")
        st.write(f"**Calculated Risk Level:** {res['risk']}")
        st.info(f"  {res['rec']}")


















