import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ----------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------
st.set_page_config(page_title="Fraud Detection App", layout="wide")

# ----------------------------------------------------------
# SAMPLE DATA
# ----------------------------------------------------------
def generate_sample_data():
    data = {
        "Customer_Name": ["Alice","Bob","Charlie","David","Evelyn","Frank",
                          "Grace","Hannah","Ivy","Jack"] * 2,
        "Transaction_Amount": np.random.randint(5000, 120000, 20),
        "Payment_Method": np.random.choice(
            ["Credit Card", "Debit Card", "UPI", "Net Banking"], 20
        ),
        "Location": np.random.choice(
            ["Mumbai", "Delhi", "Kochi", "Bangalore", "Chennai"], 20
        ),
        "Transaction_Time": [
            datetime.now() - timedelta(minutes=np.random.randint(1, 8000))
            for _ in range(20)
        ],
    }
    return pd.DataFrame(data)

# ----------------------------------------------------------
# RULE-BASED FRAUD DETECTION
# ----------------------------------------------------------
def detect_fraud(df, high_freq_limit, high_value_limit):
    df = df.copy()

    df = df.sort_values("Transaction_Time")
    df["Time_Diff"] = df["Transaction_Time"].diff().dt.total_seconds().fillna(999999)

    df["HighFreq_5min"] = (df["Time_Diff"] <= high_freq_limit).astype(int)
    df["HighValue_24H"] = df["Transaction_Amount"].apply(
        lambda x: x if x >= high_value_limit else 0
    )

    df["Fraud_Status"] = "Legit"
    df.loc[df["HighFreq_5min"] == 1, "Fraud_Status"] = "High-Freq Alert"
    df.loc[df["HighValue_24H"] > 0, "Fraud_Status"] = "High-Value Alert"

    return df

# ----------------------------------------------------------
# ML MODEL BASED FRAUD SCORE
# ----------------------------------------------------------
def generate_ml_predictions(df):
    df_ml = df.copy()
    le = LabelEncoder()

    for col in ["Customer_Name", "Payment_Method", "Location", "Fraud_Status"]:
        df_ml[col] = le.fit_transform(df_ml[col])

    X = df_ml[
        ["Transaction_Amount", "HighFreq_5min", "HighValue_24H",
         "Customer_Name", "Payment_Method", "Location"]
    ]
    y = df_ml["Fraud_Status"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25
    )

    model = RandomForestClassifier(n_estimators=200)
    model.fit(X_train, y_train)

    fraud_prob = model.predict_proba(X)[:, 1] * 100
    fraud_pred = model.predict(X)

    df["ML_Fraud_Score"] = fraud_prob.round(2)
    df["ML_Prediction"] = fraud_pred
    df["ML_Prediction"] = df["ML_Prediction"].replace({
        0: "Legit", 1: "Fraud"
    })

    return df

# ----------------------------------------------------------
# LOGIN PAGE
# ----------------------------------------------------------
def login_page():
    st.title("🔐 Secure Login")

    st.info("Use **admin/admin123** for demo access.")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == "admin" and password == "admin123":
            st.session_state["logged_in"] = True
            st.success("Login successful — welcome, admin!")
        else:
            st.error("Invalid username or password")

# ----------------------------------------------------------
# DASHBOARD PAGE
# ----------------------------------------------------------
def dashboard(df):

    st.title("🚨 Fraud Detection Output")

    # -----------------------------
    # Filters
    # -----------------------------
    st.sidebar.subheader("Filters")

    cust_filter = st.sidebar.multiselect("Customer", sorted(df["Customer_Name"].unique()))
    fraud_filter = st.sidebar.multiselect("Fraud Status", sorted(df["Fraud_Status"].unique()))
    loc_filter = st.sidebar.multiselect("Location", sorted(df["Location"].unique()))
    pay_filter = st.sidebar.multiselect("Payment Method", sorted(df["Payment_Method"].unique()))

    filtered_df = df.copy()

    if cust_filter:
        filtered_df = filtered_df[filtered_df["Customer_Name"].isin(cust_filter)]
    if fraud_filter:
        filtered_df = filtered_df[filtered_df["Fraud_Status"].isin(fraud_filter)]
    if loc_filter:
        filtered_df = filtered_df[filtered_df["Location"].isin(loc_filter)]
    if pay_filter:
        filtered_df = filtered_df[filtered_df["Payment_Method"].isin(pay_filter)]

    # -----------------------------------------
    # Main Table
    # -----------------------------------------
    def highlight_fraud(row):
        if row["Fraud_Status"] != "Legit":
            return ['background-color: #ffdddd'] * len(row)
        return [''] * len(row)

    st.dataframe(
        filtered_df.style.apply(highlight_fraud, axis=1),
        height=420,
        use_container_width=True
    )

    # ----------------------------------------------------------
    # FRAUD STATUS COUNTS
    # ----------------------------------------------------------
    st.subheader("📊 Fraud Status Counts")

    fig1 = px.bar(
        filtered_df["Fraud_Status"].value_counts(),
        title="Fraud Status Distribution"
    )
    st.plotly_chart(fig1, use_container_width=True)

    # ----------------------------------------------------------
    # TRANSACTION AMOUNT BY FRAUD STATUS (NEW)
    # ----------------------------------------------------------
    st.subheader("💰 Transaction Amount by Fraud Status")

    tab1, tab2, tab3 = st.tabs(["📦 Box Plot", "📊 Bar Chart", "🎻 Violin Plot"])

    with tab1:
        fig_box = px.box(
            filtered_df,
            x="Fraud_Status",
            y="Transaction_Amount",
            color="Fraud_Status",
            title="Transaction Amount Distribution by Fraud Status"
        )
        st.plotly_chart(fig_box, use_container_width=True)

    with tab2:
        avg_df = filtered_df.groupby("Fraud_Status")["Transaction_Amount"].mean().reset_index()
        fig_bar = px.bar(
            avg_df,
            x="Fraud_Status",
            y="Transaction_Amount",
            text="Transaction_Amount",
            color="Fraud_Status",
            title="Average Transaction Amount by Fraud Status"
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with tab3:
        fig_violin = px.violin(
            filtered_df,
            x="Fraud_Status",
            y="Transaction_Amount",
            box=True,
            points="all",
            color="Fraud_Status",
            title="Violin Plot: Transaction Amount by Fraud Status"
        )
        st.plotly_chart(fig_violin, use_container_width=True)

    # ----------------------------------------------------------
    # ML FRAUD SCORE ANALYSIS
    # ----------------------------------------------------------
    st.subheader("🤖 ML Predicted Fraud Score")

    fig4 = px.histogram(
        filtered_df,
        x="ML_Fraud_Score",
        nbins=20,
        title="ML Fraud Score Distribution"
    )
    st.plotly_chart(fig4, use_container_width=True)

    # ----------------------------------------------------------
    # DOWNLOAD BUTTON
    # ----------------------------------------------------------
    filtered_df.to_excel("fraud_detection_results.xlsx", index=False)

    with open("fraud_detection_results.xlsx", "rb") as f:
        st.download_button("⬇ Download Results", f, "fraud_detection_results.xlsx")

# ----------------------------------------------------------
# MAIN APP FLOW
# ----------------------------------------------------------
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

if not st.session_state["logged_in"]:
    login_page()
    st.stop()

# ----------------------------------------------------------
# DATA LOADING & PROCESSING
# ----------------------------------------------------------
st.sidebar.header("Options")

use_sample = st.sidebar.checkbox("Use Sample Data", True)
high_freq_limit = st.sidebar.slider("High-Frequency Threshold (sec)", 30, 600, 300)
high_value_limit = st.sidebar.slider("High-Value Threshold", 20000, 150000, 50000)

uploaded_file = st.sidebar.file_uploader("Upload Excel/CSV File", type=["csv", "xlsx"])

if use_sample:
    df = generate_sample_data()
else:
    if uploaded_file is None:
        st.warning("Upload a file or enable sample data!")
        st.stop()
    df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)

df = detect_fraud(df, high_freq_limit, high_value_limit)
df = generate_ml_predictions(df)

# Final Dashboard
dashboard(df)
