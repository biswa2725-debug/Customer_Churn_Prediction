import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score

# ==========================================================
# PAGE CONFIG
# ==========================================================
st.set_page_config(
    page_title="Customer Churn Prediction",
    layout="wide"
)

# ==========================================================
# LOAD SAVED MODEL FILES
# ==========================================================
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

# ==========================================================
# CUSTOM STYLING
# ==========================================================
st.markdown("""
<style>
.main {
    background-color: #f5f7fa;
}
.stButton>button {
    background-color: #1E90FF;
    color: white;
    font-size: 18px;
    border-radius: 10px;
    height: 3em;
    width: 100%;
}
</style>
""", unsafe_allow_html=True)

# ==========================================================
# TITLE
# ==========================================================
st.title("📊 Telecom Customer Churn Prediction System")
st.write("This AI model predicts whether a telecom customer is likely to churn.")

# ==========================================================
# NAVIGATION
# ==========================================================
page = st.radio(
    "Select Page",
    ["Predict Customer Churn", "Model Performance Dashboard"],
    horizontal=True
)

# ==========================================================
# PAGE 1 — PREDICTION
# ==========================================================
if page == "Predict Customer Churn":

    st.header("Enter Customer Details")

    # Create two columns for better layout
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior = st.selectbox("Senior Citizen", [0, 1])
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        tenure = st.slider("Tenure (Months)", 0, 72, 12)

    with col2:
        monthly = st.number_input("Monthly Charges", 0.0, 200.0, 50.0)
        total = st.number_input("Total Charges", 0.0, 10000.0, 500.0)
        contract = st.selectbox(
            "Contract Type",
            ["Month-to-month", "One year", "Two year"]
        )
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )

    # ------------------------------
    # PREPARE INPUT DATA
    # ------------------------------
    input_dict = {
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total
    }

    input_df = pd.DataFrame([input_dict])
    input_df = pd.get_dummies(input_df)
    input_df = input_df.reindex(columns=columns, fill_value=0)

    input_scaled = scaler.transform(input_df)

    # ------------------------------
    # PREDICT BUTTON
    # ------------------------------
    if st.button("🔍 Predict Churn"):

        prediction = model.predict(input_scaled)
        probability = model.predict_proba(input_scaled)[0][1]

        st.subheader("Prediction Result")

        if probability < 0.3:
            risk = "Low Risk"
            color = "green"
        elif probability < 0.6:
            risk = "Medium Risk"
            color = "orange"
        else:
            risk = "High Risk"
            color = "red"

        if prediction[0] == 1:
            st.error("⚠️ Customer is Likely to Churn")
        else:
            st.success("✅ Customer is Not Likely to Churn")

        st.markdown(f"### Probability of Churn: {probability:.2f}")
        st.markdown(f"### Risk Level: :{color}[{risk}]")
        st.progress(int(probability * 100))
        st.info("Model Used: Logistic Regression")


# ==========================================================
# PAGE 2 — MODEL PERFORMANCE DASHBOARD
# ==========================================================
elif page == "Model Performance Dashboard":

    st.header("📊 Model Performance Dashboard")

    # Load dataset
    df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

    # Preprocessing (same as training)
    df['TotalCharges'] = df['TotalCharges'].replace(' ', '0')
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'])

    df_ml = df.drop('customerID', axis=1)
    df_ml['Churn'] = df_ml['Churn'].map({'Yes': 1, 'No': 0})

    df_ml = pd.get_dummies(df_ml, drop_first=True)

    X = df_ml.drop('Churn', axis=1)
    y = df_ml['Churn']

    X = X.reindex(columns=columns, fill_value=0)
    X_scaled = scaler.transform(X)

    y_pred = model.predict(X_scaled)
    y_prob = model.predict_proba(X_scaled)[:, 1]

    # ------------------------------
    # Metrics Section
    # ------------------------------
    acc = accuracy_score(y, y_pred)
    fpr, tpr, _ = roc_curve(y, y_prob)
    roc_auc = auc(fpr, tpr)

    col1, col2 = st.columns(2)
    col1.metric("Model Accuracy", f"{acc:.2f}")
    col2.metric("AUC Score", f"{roc_auc:.2f}")

    # ------------------------------
    # Confusion Matrix
    # ------------------------------
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y, y_pred)

    fig1, ax1 = plt.subplots()
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Stayed", "Churned"],
        yticklabels=["Stayed", "Churned"]
    )
    ax1.set_xlabel("Predicted")
    ax1.set_ylabel("Actual")
    st.pyplot(fig1)

    # ------------------------------
    # ROC Curve
    # ------------------------------
    st.subheader("ROC Curve")

    fig2, ax2 = plt.subplots()
    ax2.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    ax2.plot([0, 1], [0, 1], '--')
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.legend()
    st.pyplot(fig2)

    # ------------------------------
    # Feature Importance
    # ------------------------------
    st.subheader("Top Factors Affecting Churn")

    weights = pd.Series(model.coef_[0], index=columns)
    weights = weights.sort_values()

    top_features = pd.concat([
        weights.head(10),   # Negative impact (retain)
        weights.tail(10)    # Positive impact (churn)
    ])

    # Color mapping (Positive = Blue, Negative = Red)
    colors = ['#d62728' if val < 0 else '#1f77b4' for val in top_features]

    fig3, ax3 = plt.subplots(figsize=(8, 6))
    top_features.plot(kind="barh", color=colors, ax=ax3)

    ax3.set_xlabel("Effect Size (Positive=Churn, Negative=Retain)")
    ax3.set_ylabel("Feature")

    st.pyplot(fig3)

    # ------------------------------
    # Actual Churn by Contract Type
    # ------------------------------
    st.subheader("Actual Churn by Contract Type")

    # Create grouped counts
    contract_churn = pd.crosstab(df['Contract'], df['Churn'])

    fig4, ax4 = plt.subplots(figsize=(8, 6))
    contract_churn.plot(
        kind="bar",
        ax=ax4,
        color=["#66c2a5", "#fc8d62"]  # Green = No, Orange = Yes
    )

    ax4.set_xlabel("Contract")
    ax4.set_ylabel("Count")
    ax4.legend(title="Churn")

    st.pyplot(fig4)