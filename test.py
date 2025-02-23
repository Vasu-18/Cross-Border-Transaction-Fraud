import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from fpdf import FPDF
import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
file_path = "test.csv"
data = pd.read_csv(file_path)

# Keep Customer Details for Reference
customer_details = data.copy()

# Drop Unnecessary Columns
drop_cols = ["Transaction ID", "Customer Name", "Customer Email", "Phone Number", "IP Address"]
data = data.drop(columns=drop_cols)

# Encode Categorical Variables
label_encoders = {}
for col in data.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Splitting dataset
X = data.drop(columns=["Fraud"])
y = data["Fraud"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizing Data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model Training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
fraud_cases = customer_details[customer_details["Fraud"] == True]

# ---------------- STREAMLIT APP ---------------- #
st.title("NitiSuraksha: Cross-Border Transaction Fraud Detection")

# User Input: Transaction ID or Customer ID
user_input = st.text_input("Enter Transaction ID or Customer ID:")


def generate_fraud_report(transaction, report_filename):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    
    pdf.set_font("Arial", style="B", size=16)
    pdf.cell(200, 10, "Fraud Detection Report", ln=True, align="C")
    pdf.ln(10)
    
    pdf.set_font("Arial", size=12)

    for index, row in transaction.iterrows():
        if 'Transaction ID' in row:
            pdf.cell(200, 10, f"Transaction ID: {row['Transaction ID']}", ln=True)
        if 'Customer Name' in row:
            pdf.multi_cell(0, 8, f"Customer Name: {row['Customer Name']}")
        if 'Origin Country' in row:
            pdf.multi_cell(0, 8, f"Origin Country: {row['Origin Country']}")
        if 'Destination Country' in row:
            pdf.multi_cell(0, 8, f"Destination Country: {row['Destination Country']}")
        if 'Transaction Amount' in row and 'Currency Used' in row:
            pdf.multi_cell(0, 8, f"Transaction Amount: {row['Transaction Amount']} {row['Currency Used']}")
        if 'Transaction Type' in row:
            pdf.multi_cell(0, 8, f"Transaction Type: {row['Transaction Type']}")
        if 'Merchant' in row:
            pdf.multi_cell(0, 8, f"Merchant: {row['Merchant']}")
        if 'Customer Email' in row:
            pdf.multi_cell(0, 8, f"Customer Email: {row['Customer Email']}")
        if 'Customer Age' in row:
            pdf.multi_cell(0, 8, f"Customer Age: {row['Customer Age']}")
        if 'Customer Gender' in row:
            pdf.multi_cell(0, 8, f"Customer Gender: {row['Customer Gender']}")
        if 'Device Type' in row:
            pdf.multi_cell(0, 8, f"Device Type: {row['Device Type']}")
        
        pdf.ln(10)

        # Visualization: Transaction Amount Bar Plot
        plt.figure(figsize=(8, 4))
        sns.barplot(x=["Transaction Amount"], y=[row["Transaction Amount"]])
        plt.title("Transaction Amount")
        plt.savefig("transaction_amount.png")
        plt.close()

        pdf.image("transaction_amount.png", x=10, y=None, w=190)

    # **Ensure UTF-8 encoding**
    pdf.output(report_filename, "F").encode('utf-8')
    

# Fraud Detection Process
if st.button("Check Fraud Status"):
    if user_input.strip() == "":
        st.warning("Please enter a valid Transaction ID or Customer ID.")
    else:
        transaction = customer_details[
            (customer_details["Transaction ID"].astype(str) == user_input) |
            (customer_details["Customer ID"].astype(str) == user_input)
        ]

        if transaction.empty:
            st.error("No transaction found with this ID.")
        else:
            transaction_data = transaction.drop(columns=drop_cols + ["Fraud"])
            for col in label_encoders:
                transaction_data[col] = label_encoders[col].transform(transaction_data[col])
            transaction_scaled = scaler.transform(transaction_data)
            prediction = model.predict(transaction_scaled)
            
            if prediction[0] == 1:
                st.error("🚨 Fraud Detected! 🚨")
                st.write("**Customer Details:**")
                st.dataframe(transaction)

                # Generate and Provide Fraud Report for the specific transaction
                report_file = f"fraud_report_{user_input}.pdf"
                generate_fraud_report(transaction, report_file)
                with open(report_file, "rb") as file:
                    st.download_button(label="Download Fraud Report", data=file, file_name=report_file, mime="application/pdf")
            else:
                st.success("✅ Transaction is Legitimate.")
