import streamlit as st
import pandas as pd
import joblib

# Page Configuration

st.set_page_config(
page_title="Diabetes Prediction System",
page_icon="🩺",
layout="wide"
)

# Load Model and Scaler

model = joblib.load("diabetes_model.pkl")
scaler = joblib.load("scaler.pkl")

# Title

st.markdown(
""" <h1 style='text-align:center; color:#4CAF50;'>
🩺 Diabetes Prediction System </h1>
""",
unsafe_allow_html=True
)

st.write(
"Predict whether a person is diabetic based on medical parameters."
)

st.markdown("---")

# Input Fields

col1, col2 = st.columns(2)

with col1:
    preg = st.number_input("Pregnancies", min_value=0)
    glucose = st.number_input("Glucose", min_value=0.0)
    bp = st.number_input("Blood Pressure", min_value=0.0)
    skin = st.number_input("Skin Thickness", min_value=0.0)

with col2:
    insulin = st.number_input("Insulin", min_value=0.0)
    bmi = st.number_input("BMI", min_value=0.0)
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0)
    age = st.number_input("Age", min_value=0)

st.markdown("---")

# Accuracy Card

st.metric(
label="Model Accuracy",
value="71.4%"
)

st.markdown("---")

# Prediction Button

if st.button("🔍 Predict"):


    data = pd.DataFrame(
    [[preg, glucose, bp, skin, insulin, bmi, dpf, age]],
    columns=[
            "Pregnancies",
            "Glucose",
            "BloodPressure",
            "SkinThickness",
            "Insulin",
            "BMI",
            "DiabetesPedigreeFunction",
            "Age"
        ]
    )

# Scale Input Data
    data_scaled = scaler.transform(data)

# Prediction
    prediction = model.predict(data_scaled)

    st.subheader("Prediction Result")

    if prediction[0] == 1:
         st.error("⚠️ Diabetic")
         st.warning(
             "The model predicts that the person is diabetic. Please consult a healthcare professional."
        )
    else:
        st.success("✅ Non-Diabetic")
        st.info(
            "The model predicts that the person is not diabetic."
        )
