import streamlit as st
import requests

# ===============================
# Config
# ===============================
API_URL = "http://api:8000/predict"

st.set_page_config(
    page_title="Loan Payback Prediction",
    page_icon="💰",
    layout="centered"
)

st.title("💰 Loan Payback Prediction")
st.write("Enter applicant details to predict loan payback probability.")

# ===============================
# Input Form
# ===============================
with st.form("loan_form"):

    id = st.number_input("Loan ID", min_value=1, step=1)

    annual_income = st.number_input(
        "Annual Income", min_value=0.0, value=30000.0
    )

    debt_to_income_ratio = st.slider(
        "Debt to Income Ratio", min_value=0.0, max_value=1.0, value=0.5
    )

    credit_score = st.slider(
        "Credit Score", min_value=300, max_value=850, value=650
    )

    loan_amount = st.number_input(
        "Loan Amount", min_value=0.0, value=50000.0
    )

    interest_rate = st.slider(
        "Interest Rate (%)", min_value=0.0, max_value=100.0, value=12.0
    )

    gender = st.selectbox(
        "Gender", ["Female", "Male", "Other"]
    )

    marital_status = st.selectbox(
        "Marital Status", ["Single", "Married", "Divorced", "Widowed"]
    )

    education_level = st.selectbox(
        "Education Level",
        ["High School", "Bachelor's", "Master's", "PhD", "Other"]
    )

    employment_status = st.selectbox(
        "Employment Status",
        ["Employed", "Self-employed", "Unemployed", "Student", "Retired"]
    )

    loan_purpose = st.selectbox(
        "Loan Purpose",
        ["Other", "Debt consolidation", "Home", "Education",
         "Vacation", "Car", "Medical", "Business"]
    )

    grade_subgrade = st.selectbox(
        "Grade & Subgrade",
        [
            "A1","A2","A3","A4","A5",
            "B1","B2","B3","B4","B5",
            "C1","C2","C3","C4","C5",
            "D1","D2","D3","D4","D5",
            "E1","E2","E3","E4","E5",
            "F1","F2","F3","F4","F5"
        ]
    )

    submit = st.form_submit_button("🔮 Predict")

# ===============================
# Prediction
# ===============================
if submit:

    payload = {
        "id": id,
        "annual_income": annual_income,
        "debt_to_income_ratio": debt_to_income_ratio,
        "credit_score": credit_score,
        "loan_amount": loan_amount,
        "interest_rate": interest_rate,
        "gender": gender,
        "marital_status": marital_status,
        "education_level": education_level,
        "employment_status": employment_status,
        "loan_purpose": loan_purpose,
        "grade_subgrade": grade_subgrade
    }

    try:
        response = requests.post(API_URL, json=payload)

        if response.status_code == 200:
            result = response.json()

            prob = result["loan_paid_back_probability"]
            pred = result["loan_paid_back_prediction"]

            st.success("Prediction Successful 🎉")

            st.metric(
                label="Loan Payback Probability",
                value=f"{prob:.2%}"
            )

            if pred == 1:
                st.success("✅ Loan is likely to be PAID BACK")
            else:
                st.error("❌ Loan is NOT likely to be paid back")

        else:
            st.error(f"API Error: {response.status_code}")
            st.json(response.json())

    except requests.exceptions.ConnectionError:
        st.error("🚫 Cannot connect to FastAPI server. Is it running?")
