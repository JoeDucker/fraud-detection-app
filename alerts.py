import streamlit as st
import time

def show_live_alerts(df):
    suspicious = df[df["Fraud_Status"] != "Legit"]

    st.subheader("🚨 Live Fraud Alerts")

    if suspicious.empty:
        st.success("No fraud detected right now.")
        return
    
    alert_placeholder = st.empty()

    for i in range(3):  # blink 3 times
        alert_placeholder.markdown(
            """
            <div style="padding:15px;border-radius:8px;
                 background-color:#ff4d4d;color:white;
                 animation: blinker 1s linear infinite;">
            <h3>⚠ FRAUD ALERT</h3>
            Suspicious transactions detected!
            </div>
            """,
            unsafe_allow_html=True
        )
        time.sleep(0.5)
        alert_placeholder.empty()
        time.sleep(0.5)

    st.warning("⚠ Suspicious Transactions Found!")
    st.dataframe(suspicious, use_container_width=True)
