import pandas as pd
import random
from datetime import datetime, timedelta

# ---------------------------------------------------------
# STEP 1: Create Sample Data
# ---------------------------------------------------------

data = {
    "Transaction_ID": [f"T{i:03}" for i in range(1, 21)],
    "Customer_Name": [
        "Alice", "Bob", "Charlie", "David", "Evelyn",
        "Frank", "Grace", "Hannah", "Ivy", "Jack",
        "Alice", "Bob", "Charlie", "David", "Evelyn",
        "Frank", "Grace", "Hannah", "Ivy", "Jack"
    ],
    "Transaction_Amount": [random.randint(100, 50000) for _ in range(20)],
    "Payment_Method": random.choices(
        ["Credit Card", "Debit Card", "UPI", "Net Banking"], k=20
    ),
    "Location": random.choices(
        ["Mumbai", "Delhi", "Chennai", "Bangalore", "Kochi"], k=20
    ),
    "Transaction_Time": [
        datetime.now() + timedelta(minutes=random.randint(0, 300))
        for _ in range(20)
    ]
}

df = pd.DataFrame(data)
df = df.sort_values("Transaction_Time")

# ---------------------------------------------------------
# STEP 2: High-frequency 5-minute detection (WORKS FOR ALL PANDAS)
# ---------------------------------------------------------

df["HighFreq_5min"] = 0

for name, group in df.groupby("Customer_Name"):
    times = group["Transaction_Time"].tolist()
    counts = []
    
    for i in range(len(times)):
        count = 1
        for j in range(i - 1, -1, -1):
            if (times[i] - times[j]).total_seconds() <= 300:
                count += 1
            else:
                break
        counts.append(count)
    
    df.loc[group.index, "HighFreq_5min"] = counts

# ---------------------------------------------------------
# STEP 3: High-value detection (simple rule)
# ---------------------------------------------------------

df["HighValue_24H"] = df.groupby("Customer_Name")["Transaction_Amount"].transform(
    lambda x: x.rolling(window=3, min_periods=1).sum()
)

# ---------------------------------------------------------
# STEP 4: Set fraud rules
# ---------------------------------------------------------

df["Fraud_Status"] = "Legit"

df.loc[df["HighFreq_5min"] >= 3, "Fraud_Status"] = "High-Freq Alert"
df.loc[df["HighValue_24H"] > 60000, "Fraud_Status"] = "High-Value Alert"

# ---------------------------------------------------------
# STEP 5: Save output
# ---------------------------------------------------------

df.to_excel("advanced_fraud_detection_results.xlsx", index=False)

print("✅ Advanced Fraud Detection Completed!")
print("📁 File saved: advanced_fraud_detection_results.xlsx")
print(df)
