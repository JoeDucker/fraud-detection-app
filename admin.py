import streamlit as st
import plotly.express as px

def admin_panel(df):
    st.title("🛠 Admin Dashboard")

    st.subheader("📊 System Overview")
    st.metric("Total Transactions", len(df))
    st.metric("Fraud Alerts", sum(df["Fraud_Status"] != "Legit"))

    st.subheader("Fraud by Location")
    fig = px.bar(df, x="Location", color="Fraud_Status")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Fraud by Payment Method")
    fig2 = px.pie(df, names="Payment_Method")
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Full Transaction Log")
    st.dataframe(df, use_container_width=True)
