import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Fraud Detection App", layout="wide")

# --------------------------------------------------
# SAMPLE DATA FUNCTION
# --------------------------------------------------
def generate_sample_data():
    data = {
        "Customer_Name": ["Alice","Bob","Charlie","David","Evelyn","Frank","Grace","Hannah","Ivy","Jack"] * 2,
        "Transaction_Amount": np.random.randint(5000, 120000, 20),
        "Payment_Method": np.random.choice(
            ["Credit Card", "Debit Card", "UPI", "Net Banking"], 20),
        "Location": np.random.choice(
            ["Mumbai", "Delhi", "Kochi", "Bangalore", "Chennai"], 20),
        "Transaction_Time": [
            datetime.now() - timedelta(minutes=np.random.randint(1, 8000))
            for _ in range(20)
        ],
    }
    return pd.DataFrame(data)

# --------------------------------------------------
# FRAUD DETECTION LOGIC
# --------------------------------------------------
def detect_fraud(df, high_freq_limit, high_value_limit):
    df = df.copy()
    df = df.sort_values("Transaction_Time")

    # Time differences
    df["Time_Diff"] = df["Transaction_Time"].diff().dt.total_seconds().fillna(999999)

    # High frequency
    df["HighFreq_5min"] = (df["Time_Diff"] <= high_freq_limit).astype(int)

    # High value
    df["HighValue_24H"] = df["Transaction_Amount"].apply(
        lambda x: x if x >= high_value_limit else 0
    )

    # Fraud Assignment
    df["Fraud_Status"] = "Legit"
    df.loc[df["HighFreq_5min"] == 1, "Fraud_Status"] = "High-Freq Alert"
    df.loc[df["HighValue_24H"] > 0, "Fraud_Status"] = "High-Value Alert"

    return df

# --------------------------------------------------
# MACHINE LEARNING MODEL
# --------------------------------------------------
def generate_ml_predictions(df):
    df_ml = df.copy()

    # Encode categorical variables
    le = LabelEncoder()
    for col in ["Customer_Name", "Payment_Method", "Location", "Fraud_Status"]:
        df_ml[col] = le.fit_transform(df_ml[col])

    # Features & Labels
    X = df_ml[["Transaction_Amount", "HighFreq_5min", "HighValue_24H",
               "Customer_Name", "Payment_Method", "Location"]]
    y = df_ml["Fraud_Status"]

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    # Model
    model = RandomForestClassifier(n_estimators=200)
    model.fit(X_train, y_train)

    # Prediction
    fraud_prob = model.predict_proba(X)[:, 1] * 100
    fraud_pred = model.predict(X)

    df["ML_Fraud_Score"] = fraud_prob.round(2)
    df["ML_Prediction"] = fraud_pred

    df["ML_Prediction"] = df["ML_Prediction"].replace({
        0: "Legit",
        1: "Fraud"
    })

    return df


# --------------------------------------------------
# SIDEBAR OPTIONS
# --------------------------------------------------
st.sidebar.header("Options")

use_sample = st.sidebar.checkbox("Use Sample Data", True)

high_freq_limit = st.sidebar.slider("High-Frequency Threshold (sec)",
                                    30, 600, 300)
high_value_limit = st.sidebar.slider("High-Value Threshold",
                                     20000, 100000, 50000)

uploaded_file = st.sidebar.file_uploader("Upload Excel/CSV", type=["csv", "xlsx"])

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
if use_sample:
    df = generate_sample_data()
else:
    if uploaded_file is None:
        st.warning("Upload a file or enable sample data!")
        st.stop()
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

# --------------------------------------------------
# PROCESS + ML
# --------------------------------------------------
df_result = detect_fraud(df, high_freq_limit, high_value_limit)
df_result = generate_ml_predictions(df_result)

# --------------------------------------------------
# FILTERS
# --------------------------------------------------
st.sidebar.subheader("Filters")

customer_filter = st.sidebar.multiselect("Customer", df_result["Customer_Name"].unique())
fraud_filter = st.sidebar.multiselect("Fraud Status", df_result["Fraud_Status"].unique())
location_filter = st.sidebar.multiselect("Location", df_result["Location"].unique())
payment_filter = st.sidebar.multiselect("Payment Method", df_result["Payment_Method"].unique())

filtered_df = df_result.copy()

if customer_filter:
    filtered_df = filtered_df[filtered_df["Customer_Name"].isin(customer_filter)]
if fraud_filter:
    filtered_df = filtered_df[filtered_df["Fraud_Status"].isin(fraud_filter)]
if location_filter:
    filtered_df = filtered_df[filtered_df["Location"].isin(location_filter)]
if payment_filter:
    filtered_df = filtered_df[filtered_df["Payment_Method"].isin(payment_filter)]

# --------------------------------------------------
# MAIN TABLE
# --------------------------------------------------
st.title("🚨 Fraud Detection Output")

def highlight_fraud(row):
    if row["Fraud_Status"] != "Legit":
        return ['background-color: #ffdddd'] * len(row)
    return [''] * len(row)

st.dataframe(filtered_df.style.apply(highlight_fraud, axis=1), height=420)

# --------------------------------------------------
# CHARTS SECTION
# --------------------------------------------------
st.subheader("📊 Fraud Status Counts")
fig1 = px.bar(filtered_df["Fraud_Status"].value_counts(),
              title="Fraud Status Distribution")
st.plotly_chart(fig1, use_container_width=True)

st.subheader("🌍 Fraud Count by Location")
fig2 = px.histogram(filtered_df, x="Location", color="Fraud_Status",
                    barmode="group")
st.plotly_chart(fig2, use_container_width=True)

st.subheader("💰 Transaction Amount Distribution")
fig3 = px.box(filtered_df, x="Fraud_Status", y="Transaction_Amount",
              color="Fraud_Status")
st.plotly_chart(fig3, use_container_width=True)

st.subheader("🤖 ML Predicted Fraud Probability")
fig4 = px.histogram(filtered_df, x="ML_Fraud_Score", nbins=20,
                    title="ML Fraud Score Distribution")
st.plotly_chart(fig4, use_container_width=True)

# --------------------------------------------------
# DOWNLOAD RESULTS
# --------------------------------------------------
df_result.to_excel("fraud_detection_results.xlsx", index=False)

with open("fraud_detection_results.xlsx", "rb") as f:
    st.download_button(
        "⬇ Download Results",
        f,
        "fraud_detection_results.xlsx"
    )
