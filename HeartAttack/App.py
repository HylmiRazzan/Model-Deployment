import streamlit as st
import joblib
import os
import base64
import numpy as np

BASE_DIR = os.path.dirname(__file__)
img_path = os.path.join(BASE_DIR, "HeartBackground.png")

if os.path.exists(img_path):
    with open(img_path, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    st.markdown(f"""
        <style>
        .stApp {{
            background: url("data:image/png;base64,{data}");
            background-size: cover;
            background-position: center;
        }}
        .block-container {{
            margin-left: 0 !important;
            margin-right: auto !important;
            padding-left: 20% !important;
            padding-top: 3% !important;
            padding-bottom: 100px !important;
            max-width: 500px !important;
            background-color: rgba(0, 0, 0, 1);
            border-radius: 0 20px 20px 0;
            min-height: 100vh !important;
            height: auto !important;
            overflow-y: auto !important;
        }}
        h1, label {{ color: white !important; }}
        </style>
        """, unsafe_allow_html=True)


model = joblib.load(os.path.join(BASE_DIR, "model.pkl"))

def main():
    st.title('Heart Attack Risk Prediction')

    gender_options = {"Male": 1, "Female": 0}
    gender_label = st.selectbox("Gender", options=list(gender_options.keys()))
    sex = gender_options[gender_label]

    age = st.number_input("Age", min_value=1, max_value=120, value= None)
    cp = st.number_input("Chest Pain", min_value=0, max_value=3, value= None)
    trestbps = st.number_input("Blood Pressure", min_value=0, max_value=200, value= None)
    chol = st.number_input("Cholesterol (mg/dL)", min_value=0, max_value=600, value= None)

    fbs_options = { "Normal": 0, "High": 1}
    fbs_label = st.selectbox("Blood Sugar", options=list(fbs_options.keys()))
    fbs = fbs_options[fbs_label]

    restecg_options = {"Normal": 0, "ST-T Abnormal": 1, "LVH": 2}
    restecg_label = st.selectbox("ECG result", options=list(restecg_options.keys()))
    resrecg = restecg_options[restecg_label]

    thalach = st.number_input("Max Heart Rate", min_value= 40, max_value=250, value= None)

    exang_options = {"No": 0, "Yes": 1}
    exang_label = st.selectbox("Exercise Angina", options=list(exang_options.keys()))
    exang = exang_options[exang_label]

    oldpeak = st.number_input("ST Depression", min_value= 0.0, max_value=7.0, value= None, step = 0.1, format="%.1f")

    slope_options = {"Upsloping (Normal)": 0, "Flat": 1, "Downsloping": 2}
    slope_label = st.selectbox("ST Slope", options=list(slope_options.keys()))
    slope = slope_options[slope_label]

    ca = st.number_input("Number of Major Vessels (0-4)", min_value=0, max_value= 4, value= None)

    thal_options = {"Fixed Defect": 1, "Normal": 2, "Reversable Defect": 3}
    thal_label = st.selectbox("Thalassemia", options=list(thal_options.keys()))
    thal = thal_options[thal_label]

    if st.button("Predict !"):
        features = [age, sex, cp, trestbps, chol, fbs, resrecg, thalach, exang, oldpeak, slope, ca, thal]
        result = make_prediction(features)
        if result == 1:
            prediction_text = "High Risk of Heart Attack"
            st.error(f'Result: {prediction_text}')
        else:
            prediction_text = "Low Risk / Normal"
            st.success(f'Result: {prediction_text}')

def make_prediction(features):
    input_array = np.array(features).reshape(1, -1)
    prediction = model.predict(input_array)
    return prediction[0]
if __name__ == '__main__':
    main()



