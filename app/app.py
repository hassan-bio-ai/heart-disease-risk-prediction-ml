import sys
from pathlib import Path

import streamlit as st


BASE_DIR = Path(__file__).resolve().parent.parent
SRC_PATH = BASE_DIR / "src"

sys.path.append(str(SRC_PATH))

from predict import predict_patient_risk


st.set_page_config(
    page_title="Heart Disease Prediction System",
    layout="centered"
)

st.title("❤️ Heart Disease Prediction System")

st.write(
    "Enter patient clinical information to estimate heart disease risk."
)

# ---------------------------
# Patient Inputs
# ---------------------------

age = st.slider("Age", 20, 100, 55)

sex = st.selectbox(
    "Sex",
    options=[0, 1],
    format_func=lambda x: "Female" if x == 0 else "Male"
)

cp = st.selectbox(
    "Chest Pain Type (cp)",
    options=[0, 1, 2, 3]
)

trestbps = st.slider("Resting Blood Pressure", 80, 220, 130)

chol = st.slider("Cholesterol", 100, 600, 250)

fbs = st.selectbox(
    "Fasting Blood Sugar > 120 mg/dl",
    options=[0, 1]
)

restecg = st.selectbox(
    "Resting ECG",
    options=[0, 1, 2]
)

thalach = st.slider("Maximum Heart Rate", 60, 220, 150)

exang = st.selectbox(
    "Exercise Induced Angina",
    options=[0, 1]
)

oldpeak = st.slider(
    "Oldpeak",
    0.0,
    6.0,
    1.0
)

slope = st.selectbox(
    "Slope",
    options=[0, 1, 2]
)

ca = st.selectbox(
    "Number of Major Vessels",
    options=[0, 1, 2, 3, 4]
)

thal = st.selectbox(
    "Thalassemia",
    options=[0, 1, 2, 3]
)

# ---------------------------
# Prediction Button
# ---------------------------

if st.button("Predict Risk"):

    patient_data = {
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal
    }

    result = predict_patient_risk(patient_data)

    st.subheader("Prediction Result")

    st.write(f"Prediction: {result['prediction']}")

    st.write(f"Risk Probability: {result['risk_percentage']}%")

    st.write(f"Risk Level: {result['risk_level']}")

    if result["risk_level"] == "Low Risk":
        st.success("Low Risk")

    elif result["risk_level"] == "Medium Risk":
        st.warning("Medium Risk")

    else:
        st.error("High Risk")