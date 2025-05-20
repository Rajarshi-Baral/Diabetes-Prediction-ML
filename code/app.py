import streamlit as st
import pandas as pd
import joblib
import os
from datetime import datetime

# Load the trained model
model_path = os.path.join("..", "dataset", "diabetes_rf_model.pkl")
model = joblib.load(model_path)

# CSV file path to log submissions
csv_path = os.path.join("..", "dataset", "model_testing.csv")

# Function to save user input
def save_input(data):
    df = pd.DataFrame([data])
    if os.path.exists(csv_path):
        df.to_csv(csv_path, mode='a', header=False, index=False)
    else:
        df.to_csv(csv_path, mode='w', header=True, index=False)

# Function to get recent submissions
def get_recent_submissions(n=5):
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        return df.tail(n)
    else:
        return pd.DataFrame()

# Streamlit UI
st.title("🩺 Diabetes Prediction App")
st.markdown("Predict whether a patient is diabetic using medical data.")

# Input fields
pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, step=1)
glucose = st.number_input("Glucose Level", min_value=0, max_value=300, step=1)
blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=200, step=1)
skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, step=1)
insulin = st.number_input("Insulin Level", min_value=0, max_value=900, step=1)
bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, step=0.1)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, step=0.01)
age = st.number_input("Age", min_value=1, max_value=120, step=1)

# Submit button
if st.button("Predict"):
    input_data = [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]
    prediction = model.predict([input_data])[0]
    result_text = "Diabetic" if prediction == 1 else "Not Diabetic"
    st.subheader("Prediction Result:")
    st.success(result_text)

    # Save data with timestamp
    input_dict = {
        'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': blood_pressure,
        'SkinThickness': skin_thickness,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': dpf,
        'Age': age,
        'Prediction': result_text
    }
    save_input(input_dict)

# Show recent submissions
st.markdown("### 🕒 Recent Submissions")
recent = get_recent_submissions()
if not recent.empty:
    st.dataframe(recent)
else:
    st.info("No submissions yet.")
