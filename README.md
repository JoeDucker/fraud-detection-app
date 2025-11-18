# 🚨 Fraud Detection Web App  
A powerful, interactive Streamlit-based Fraud Detection System that analyzes transactions, flags anomalies, predicts fraudulent activity using Machine Learning, and visualizes fraud patterns with insightful charts.

🔗 **Live App:**  
https://joeducker-fraud-detection-app-app-4ap9re.streamlit.app/

---

## 📌 **Overview**

The **Fraud Detection Web App** allows users to:

- Upload or use sample transaction datasets  
- Detect suspicious transactions using **rule-based** and **machine learning-based** fraud detection  
- View fraud alerts in an interactive dashboard  
- Filter data by customer, fraud status, location, and payment method  
- Visualize fraud patterns via charts and graphs  
- Download full fraud reports in Excel format  

The system uses a combination of **business rules** and a trained **Random Forest Machine Learning model** to evaluate fraud probability.

---

## 🚀 **Key Features**

### 🔍 1. Rule-Based Fraud Detection  
- High-frequency transaction alerts  
- High-value transaction alerts  
- Suspicious pattern detection

### 🤖 2. Machine Learning Model  
- Uses Random Forest Classifier  
- Generates:
  - `ML_Fraud_Score` (probability %)  
  - `ML_Prediction` (Legit / Fraud)

### 📊 3. Interactive Visual Dashboards  
- Fraud Status Distribution  
- Fraud by Location  
- Transaction Amount by Fraud Status  
- ML Score Distribution  

### 🧰 4. Advanced Filtering  
- Customer  
- Location  
- Payment Method  
- Fraud Status  

### 📥 5. File Upload & Download  
- Upload CSV/XLSX  
- Download fraud results as Excel  

### 🔐 6. Simple Admin Login (Optional)  
- Role-based access  
- Admin-only panel  

---

## 🛠 **Tech Stack**

| Component | Technology |
|----------|------------|
| UI Framework | Streamlit |
| ML Model | RandomForestClassifier |
| Data Processing | Pandas, NumPy |
| Visualization | Plotly Express |
| Authentication | Streamlit Session State |
| Deployment | Streamlit Cloud |

---

## 📁 **Project Structure**
fraud-detection-app/
│
├── app.py # Main Streamlit application
├── fraud_detection.py # Fraud detection logic
├── requirements.txt # Python dependencies
├── README.md # Project documentation


---

## 💻 **Run the App Locally**

### **1️⃣ Clone the repository**

git clone https://github.com/your-username/fraud-detection-app.git
cd fraud-detection-app 

### 2️⃣ **Install dependencies**

pip install -r requirements.txt

### 3️⃣ **Run Streamlit**

streamlit run app.py

