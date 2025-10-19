import streamlit as st
import joblib
import numpy as np
import pandas as pd


# --- Load trained model and preprocessing tools ---
model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")

# Load saved label encoders
le_gender = joblib.load("gender_encoder.pkl")
le_smoke = joblib.load("smoke_encoder.pkl")

# --- Streamlit UI ---
st.title("🩺 Diabetes Risk Prediction App")
st.markdown("This app predicts the likelihood of diabetes based on your health details.")

st.sidebar.header("ℹ️ About the Model")
st.sidebar.info(
    "This model was trained on medical data to predict diabetes using Logistic Regression. "
    "It takes into account glucose, HbA1c, BMI, and lifestyle-related factors."
)

# --- Input Fields ---
gender = st.selectbox("Gender", le_gender.classes_.tolist())
age = st.number_input("Age", min_value=1, max_value=120, value=35)
hypertension = st.selectbox("Hypertension (1 = Yes, 0 = No)", [0, 1])
heart_disease = st.selectbox("Heart Disease (1 = Yes, 0 = No)", [0, 1])
smoking_history = st.selectbox("Smoking History", le_smoke.classes_.tolist())
bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
HbA1c_level = st.number_input("HbA1c Level", min_value=3.0, max_value=9.0, value=5.5)
blood_glucose_level = st.number_input("Blood Glucose Level", min_value=50, max_value=300, value=100)

# --- Encode categorical features using trained encoders ---
gender_num = le_gender.transform([gender])[0]
smoking_num = le_smoke.transform([smoking_history])[0]

# --- Create input dataframe ---
columns = [
    "gender", "age", "hypertension", "heart_disease", "smoking_history",
    "bmi", "HbA1c_level", "blood_glucose_level"
]
input_data = pd.DataFrame([[gender_num, age, hypertension, heart_disease,
                            smoking_num, bmi, HbA1c_level, blood_glucose_level]],
                          columns=columns)

# --- Scale numerical data ---
input_scaled = scaler.transform(input_data)
print(input_data)
# --- Prediction ---
if st.button("🔍 Predict Diabetes Risk"):
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    print(prediction, probability)
    st.subheader("📊 Prediction Result")
    if prediction >= 0.8:
        st.error(f"⚠️ **High Risk of Diabetes**\nProbability: {probability:.2f}")
    else:
        st.success(f"✅ **Low Risk of Diabetes**\nProbability: {probability:.2f}")

    # Show probability breakdown
    st.write("### Model Probability Distribution")
    st.json({
        "Low Risk (0)": float(model.predict_proba(input_scaled)[0][0]),
        "High Risk (1)": float(model.predict_proba(input_scaled)[0][1])
    })
