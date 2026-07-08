import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model and preprocessing tools
model = joblib.load("../data/final_model.pkl")
imputer = joblib.load("../data/imputer.pkl")
scaler = joblib.load("../data/scaler.pkl")

FEATURES = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]

st.set_page_config(page_title="Diabetes Risk Predictor", page_icon="🩺", layout="centered")

st.title("🩺 Diabetes Risk Predictor")
st.write("Enter patient medical data below to estimate diabetes risk.")

col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
    glucose = st.number_input("Glucose", min_value=0, max_value=300, value=120)
    blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=200, value=70)
    skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)

with col2:
    insulin = st.number_input("Insulin", min_value=0, max_value=900, value=80)
    bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)
    age = st.number_input("Age", min_value=1, max_value=120, value=30)

if st.button("Predict Risk", type="primary"):
    input_data = pd.DataFrame([[pregnancies, glucose, blood_pressure, skin_thickness,
                                 insulin, bmi, dpf, age]], columns=FEATURES)

    # Same preprocessing as training: 0 -> NaN for medical columns, then impute + scale
    cols_to_fix = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    input_data[cols_to_fix] = input_data[cols_to_fix].replace(0, np.nan)

    input_imputed = pd.DataFrame(imputer.transform(input_data), columns=FEATURES)
    input_scaled = pd.DataFrame(scaler.transform(input_imputed), columns=FEATURES)

    proba = model.predict_proba(input_scaled)[0][1]
    prediction = model.predict(input_scaled)[0]

    st.divider()
    if prediction == 1:
        st.error(f"⚠️ High risk of diabetes — estimated probability: {proba:.1%}")
    else:
        st.success(f"✅ Low risk of diabetes — estimated probability: {proba:.1%}")

    st.progress(float(proba))
    st.caption("This tool is for educational purposes only and does not replace professional medical diagnosis.")