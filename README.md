# End-to-End Payment Fraud Detection System using K-Means Clustering

This repository contains a full-stack Machine Learning project designed to detect fraudulent transactions in luxury cosmetics retail. Built as a final project for the Introduction to Machine Learning course at Parami University.

Presentation Slides: https://canva.link/ytddgqkj1w0co4e 

# 🚀 Live Demo

Check out the interactive web application here: https://fraud-detect17.streamlit.app/

# 📊 Project Overview

This project uses unsupervised machine learning (K-Means Clustering) to segment incoming customer transactions into 4 distinct risk personas. By analyzing behavioral features like purchase amounts, time of day, and loyalty status, the system provides real-time decision support for loss prevention teams.

The model classifies transactions into the following actionable clusters:

• Standard Everyday Buyer (Low Risk): Typical shopping patterns. Automated logic auto-approves these transactions instantly.

• High-Value Alert (High Risk): Large purchase amount outliers. Blocks transaction on immediate hold until phone identity verification is complete.

• Late-Night Anomaly (Medium Risk): Transactions occurring at unusual hours. Flags the order for manual review to cross-reference IP and shipping addresses.

• Verified VIP Client (Safe): Trusted loyalty members. Grants priority handling and applies complimentary express shipping.

The final trained clustering model is serialized via Joblib and served through an interactive Streamlit dashboard to let operations teams input parameters and immediately view the corresponding risk action and color-coded alert.

# 📁 Repository Structure

• app.py: The Streamlit web application interface and user inputs.

• fraud_detect.py: Data preprocessing pipeline and model prediction logic.

• fraud_detection_Final_model.pkl: Saved K-Means clustering model.

• luxury_cosmetics_fraud_analysis_2025.csv: The dataset used for training and analysis.

• requirements.txt: Python package dependencies needed to run the environment.

# 🛠️ Tech Stack

• Language: Python

• Machine Learning: Scikit-Learn (K-Means)

• Deployment: Streamlit

• Model Management: Joblib / Pickle
