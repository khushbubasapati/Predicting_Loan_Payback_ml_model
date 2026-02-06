import streamlit as st
import requests

# ===============================
# Config
# ===============================
API_URL = "http://api:8000/predict"

st.set_page_config(
    page_title="Loan Payback Predictor",
    page_icon="💰",
    layout="centered"
)

# ===============================
# CUSTOM CSS
# ===============================
st.markdown("""
<style>
[data-baseweb="select"] > div {
    border-right: 0px !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
/* REMOVE STREAMLIT LINES */
hr {display:none!important;}
[data-testid="stVerticalBlock"] > div {border:none!important;}

/* BACKGROUND */
html, body, .stApp {
    background: radial-gradient(circle at top,#0f172a,#020617) !important;
    color: white !important;
}

/* GLOW ANIMATION */
@keyframes glow {
    0% {background-position:0%}
    100% {background-position:300%}
}

/* TITLE GRADIENT */
.grad {
    background: linear-gradient(90deg,#6366f1,#22d3ee,#ec4899,#6366f1);
    background-size:300%;
    animation: glow 6s linear infinite;
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
}

/* LABELS */
label {
    color:white!important;
    font-weight:600!important;
}

/* FORM CARD */
[data-testid="stForm"] {
    background:#020617!important;
    padding:30px;
    border-radius:24px;
}

/* INPUTS */
input, select {
    background:#020617!important;
    color:white!important;
    border-radius:14px!important;
    border:1px solid #334155!important;
}

/* SLIDERS */
[data-baseweb="slider"] span {
    color:white!important;
}

/* BUTTON */
button {
    background:linear-gradient(90deg,#6366f1,#22d3ee)!important;
    border-radius:16px!important;
    font-weight:700!important;
}

/* RESULT */
.result-box {
    margin-top:25px;
    background:#020617;
    padding:25px;
    border-radius:22px;
    text-align:center;
    box-shadow:0 0 40px rgba(99,102,241,.4);
}

.big-prob {
    font-size:60px;
    font-weight:900;
}

.good {color:#22c55e;font-size:24px;font-weight:700;}
.bad {color:#ef4444;font-size:24px;font-weight:700;}

/* KILL ALL HORIZONTAL SEPARATORS */
[data-testid="stVerticalBlockBorderWrapper"] {
    border: none !important;
}

[data-testid="stWidget"] {
    border: none !important;
}

/* REMOVE SHADOW LINES */
[data-testid="stVerticalBlock"] {
    box-shadow: none !important;
}

/* CLEAN SLIDERS */
[data-baseweb="slider"] > div {
    box-shadow:none!important;
    background:transparent!important;
}

/* HIDE TEXT CURSOR */
input, select {
    caret-color: transparent !important;
}

</style>
""", unsafe_allow_html=True)


# ===============================
# TITLE
# ===============================
st.markdown("""
<h1 style="text-align:center;font-weight:900;font-size:48px">
<span>💰</span><span class="grad"> Loan Payback Predictor</span>
</h1>
<p style="text-align:center">✨ AI-powered credit risk assessment</p>
""", unsafe_allow_html=True)

# ===============================
# FORM
# ===============================
with st.form("loan_form"):

    annual_income = st.number_input("Annual Income", 0.0, value=30000.0)
    loan_amount = st.number_input("Loan Amount", 0.0, value=5000.0)

    credit_score = st.slider("Credit Score", 300, 850, 650)
    debt_ratio = st.slider("Debt-to-Income Ratio", 0.0, 1.0, 0.4)
    interest = st.slider("Interest Rate %", 0.0, 100.0, 12.0)

    marital = st.selectbox("Marital Status",["Single","Married","Divorced","Widowed"])
    gender = st.selectbox("Gender",["Female","Male","Other"])
    employment = st.selectbox("Employment",["Employed","Self-employed","Unemployed","Student","Retired"])
    education = st.selectbox("Education",["High School","Bachelor's","Master's","PhD","Other"])
    purpose = st.selectbox("Loan Purpose",["Other","Debt","Home","Education","Car","Medical","Business"])
    grade = st.selectbox("Credit Grade",["A1","A2","B1","B2","C1","C2","D1","D2","E1","F1"])

    submit = st.form_submit_button("🔮 Predict")

# ===============================
# PREDICTION
# ===============================
if submit:

    payload = {
        "annual_income": annual_income,
        "debt_to_income_ratio": debt_ratio,
        "credit_score": credit_score,
        "loan_amount": loan_amount,
        "interest_rate": interest,
        "gender": gender,
        "marital_status": marital,
        "education_level": education,
        "employment_status": employment,
        "loan_purpose": purpose,
        "grade_subgrade": grade
    }

    try:
        r = requests.post(API_URL,json=payload)

        if r.status_code==200:

            res=r.json()
            prob=res["loan_paid_back_probability"]
            pred=res["loan_paid_back_prediction"]

            st.markdown(f"""
            <div class="result-box">
                <div>{"Loan Payback Probability"}</div>
                <div class="big-prob">{prob:.2%}</div>
            """,unsafe_allow_html=True)

            if pred==1:
                st.markdown('<div class="good">✅ LOW RISK – Loan likely PAID BACK</div>',unsafe_allow_html=True)
            else:
                st.markdown('<div class="bad">❌ HIGH RISK – Loan likely DEFAULT</div>',unsafe_allow_html=True)

            st.markdown("</div>",unsafe_allow_html=True)

        else:
            st.error("API Error")

    except:
        st.error("🚫 FastAPI not reachable")
