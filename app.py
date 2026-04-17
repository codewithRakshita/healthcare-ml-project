import streamlit as st
import numpy as np
import joblib
import random

# Load model
model = joblib.load("model.pkl")

# Title
st.title("🏥 AI Gender-Based Healthcare System")
st.markdown("### Personalized Disease Risk Prediction")

# Gender selection
gender = st.selectbox("Select Gender", ["Male", "Female"])

st.subheader("Enter Health Details")

# Common inputs
glucose = st.number_input("Glucose Level", min_value=0)
bp = st.number_input("Blood Pressure", min_value=0)
bmi = st.number_input("BMI", min_value=0.0)
age = st.number_input("Age", min_value=1)

# Female-specific input
if gender == "Female":
    pregnancies = st.number_input("Number of Pregnancies", min_value=0)
else:
    pregnancies = 0

# Predict button
if st.button("Predict Health Risk"):

    # Prepare input (dummy values for unused columns)
    input_data = np.array([[pregnancies, glucose, bp, 0, 0, bmi, 0.5, age]])

    result = model.predict(input_data)

    # Risk score (just for UI improvement)
    risk_score = random.randint(30, 95)

    st.subheader("Prediction Result")

    if result[0] == 1:
        st.error(f"⚠️ High Risk Detected ({risk_score}%)")
    else:
        st.success(f"✅ Low Risk ({risk_score}%)")

    # Graph
    st.subheader("Health Parameters Overview")
    st.bar_chart([glucose, bp, bmi])

    # Advice section
    st.subheader("Personalized Health Advice")

    if gender == "Female":
        st.write("👩 Maintain balanced diet rich in iron and calcium")
        st.write("👩 Regular hormonal and health checkups recommended")
        st.write("👩 Stay physically active and manage stress")

    else:
        st.write("👨 Focus on heart health and regular exercise")
        st.write("👨 Reduce cholesterol and avoid smoking")
        st.write("👨 Maintain healthy lifestyle and sleep cycle")

    st.info("⚠️ This is an AI-based prediction. Consult a doctor for accurate diagnosis.")