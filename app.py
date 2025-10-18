import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load saved model and scaler
model = joblib.load('diabetes_model.pkl')
scaler = joblib.load('scaler.pkl')

# App title
st.title("🩺 Diabetes Risk Prediction App")
st.markdown("Enter the details below to check the likelihood of diabetes.")

# Sidebar info
st.sidebar.header("About")
st.sidebar.info("This app uses a machine learning model trained on medical data to predict diabetes risk.")

# Input fields
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", min_value=1, max_value=120, value=30)
hypertension = st.selectbox("Hypertension", [0, 1])
heart_disease = st.selectbox("Heart Disease", [0, 1])
smoking_history = st.selectbox("Smoking History", ["never", "former", "current", "No Info"])
bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
HbA1c_level = st.number_input("HbA1c Level", min_value=3.0, max_value=9.0, value=5.5)
blood_glucose_level = st.number_input("Blood Glucose Level", min_value=50, max_value=300, value=100)

# Encoding categorical values manually (same logic used during training)
gender_num = 1 if gender == "Male" else 0

# Smoking encoding (match what was done during preprocessing)
if smoking_history == "never":
    smoking_num = 0
elif smoking_history == "former":
    smoking_num = 1
elif smoking_history == "current":
    smoking_num = 2
else:
    smoking_num = 3


# Create input array
input_data = np.array([[gender_num, age, hypertension, heart_disease, smoking_num,
                        bmi, HbA1c_level, blood_glucose_level]])
columns = ['gender', 'age', 'hypertension', 'heart_disease', 
           'smoking_history', 'bmi', 'HbA1c_level', 'blood_glucose_level']

input_df = pd.DataFrame(input_data, columns=columns)
# Scale it
input_scaled = scaler.transform(input_df)

# Predict
if st.button("🔍 Predict Diabetes Risk"):
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]  # Probability of being diabetic

    if prediction == 1:
        st.error(f"⚠️ High Risk of Diabetes (Probability: {probability:.2f})")
    else:
        st.success(f"✅ Low Risk of Diabetes (Probability: {probability:.2f})")
