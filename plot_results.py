import pandas as pd
import matplotlib.pyplot as plt

# load results
df = pd.read_excel("advanced_fraud_detection_results.xlsx")

# 1) Fraud status counts
counts = df["Fraud_Status"].value_counts()
plt.figure(figsize=(6,4))
counts.plot(kind="bar")
plt.title("Fraud Status Counts")
plt.xlabel("Fraud Status")
plt.ylabel("Count")
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig("fraud_status_counts.png")
plt.show()

# 2) Top customers by alerts
alerts = df[df["Fraud_Status"] != "Legit"]
top = alerts["Customer_Name"].value_counts()
plt.figure(figsize=(7,4))
top.plot(kind="bar")
plt.title("Alerts by Customer (non-legit)")
plt.xlabel("Customer")
plt.ylabel("Number of Alerts")
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig("alerts_by_customer.png")
plt.show()
