import streamlit as st
import joblib
import pandas as pd

model=joblib.load("model.pkl")
sc=joblib.load("scaler.pkl")
expected_columns= joblib.load("columns.pkl")


st.title("Heart Disease Prediction 🫀")
st.markdown("Provide following Details 👇")

age=st.slider("Age",18,100,40)
sex=st.selectbox("Sex",['M','F'])
painType=st.selectbox("Chest Pain Type",[1,2,3,4])
blood_pressure=st.number_input("Blood Pressure(mm/hg)", 80,200,120)
cholesterol=st.number_input("Cholesterol(mg/dl)",100,600,200)
Fasting_bs=st.selectbox("Fasting Blood Suger >120 mg/dl",[0,1])
Ecg=st.selectbox("ECG",{0,1,2})
max_hr = st.number_input(
    "Max Heart Rate Achieved",
    min_value=50,
    max_value=250,
    value=150
)

exercise_angina = st.selectbox(
    "Exercise Induced Angina",
    ["No", "Yes"]
)

st_depression = st.number_input(
    "ST Depression",
    min_value=0.0,
    max_value=10.0,
    value=1.0,
    step=0.1
)

slope_st = st.selectbox(
    "Slope of ST Segment",
    [0, 1, 2]
)

num_vessels = st.selectbox(
    "Number of Major Vessels Fluoroscopy",
    [0, 1, 2, 3, 4]
)

thallium = st.selectbox(
    "Thallium Test Result",
    [0, 1, 2, 3]
)
sex = 1 if sex == 'M' else 0
exercise_angina = 1 if exercise_angina == 'Yes' else 0

if st.button("Predict"):

    raw_input = {
        'Age': age,
        'Sex': sex,
        'Chest pain type': painType,
        'BP': blood_pressure,
        'Cholesterol': cholesterol,
        'FBS over 120': Fasting_bs,
        'EKG results': Ecg,
        'Max HR': max_hr,
        'Exercise angina': exercise_angina,
        'ST depression': st_depression,
        'Slope of ST': slope_st,
        'Number of vessels fluro': num_vessels,
        'Thallium': thallium
    }

    input_df = pd.DataFrame([raw_input])

    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[expected_columns]

    scaled_input = sc.transform(input_df)

    prediction = model.predict(scaled_input)[0]

    if prediction == 1:
        st.error("🫀 High Risk of Heart Disease")
    else:
        st.success("👌 Low Risk of Heart Disease")