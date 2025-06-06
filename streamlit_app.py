import streamlit as st
import pandas as pd
import joblib

# Load models and columns
logreg_model = joblib.load("model_logreg.pkl")
tree_model = joblib.load("model_tree.pkl")
columns = joblib.load("columns.pkl")

# UI
st.title("E-commerce Customer Churn Prediction")

# User inputs
def get_user_input():
    data = {
        'Tenure': st.slider("Tenure", 0, 60, 12),
        'WarehouseToHome': st.slider("Distance to Warehouse", 1, 50, 10),
        'NumberOfDeviceRegistered': st.slider("Devices Registered", 1, 10, 3),
        'PreferedOrderCat': st.selectbox("Preferred Order Category", ['Laptop & Accessory', 'Mobile', 'Fashion', 'Others']),
        'SatisfactionScore': st.slider("Satisfaction Score", 1, 5, 3),
        'MaritalStatus': st.selectbox("Marital Status", ['Single', 'Married', 'Divorced']),
        'NumberOfAddress': st.slider("Number of Addresses", 1, 10, 2),
        'Complain': st.selectbox("Complaint Filed?", [0, 1]),
        'DaySinceLastOrder': st.slider("Days Since Last Order", 0, 30, 7),
        'CashbackAmount': st.slider("Avg Cashback", 0, 1000, 200)
    }
    return pd.DataFrame([data])

input_df = get_user_input()

# Encode categorical columns manually (match training)
encode_map = {
    'PreferedOrderCat': {'Laptop & Accessory': 0, 'Mobile': 1, 'Fashion': 2, 'Others': 3},
    'MaritalStatus': {'Single': 2, 'Married': 1, 'Divorced': 0}
}

for col in encode_map:
    input_df[col] = input_df[col].map(encode_map[col])

# Reorder columns
input_df = input_df[columns]

# Predictions
if st.button("Predict Churn"):
    logreg_pred = logreg_model.predict(input_df)[0]
    tree_pred = tree_model.predict(input_df)[0]

    st.subheader("Predictions")
    st.write(f"🔹 Logistic Regression: {'Churn' if logreg_pred == 1 else 'No Churn'}")
    st.write(f"🔹 Decision Tree: {'Churn' if tree_pred == 1 else 'No Churn'}")
