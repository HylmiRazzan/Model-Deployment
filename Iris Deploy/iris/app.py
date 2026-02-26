import streamlit as st
import joblib
import numpy as np
import pandas as pd


# Load preprocess and model from MLflow
# Load preprocessor
scaler = joblib.load("preprocessor.pkl")
model = joblib.load("model.pkl")

def main():
    st.title('Machine Learning Iris Prediction Model Deployment')

    # Add user input components for 5 features
    sepal_length = st.number_input('input nilai sepal_length', min_value=0.0, max_value=10.0, value=0.1)
    sepal_width = st.number_input('sepal_width', min_value=0.0, max_value=10.0, value=0.1)
    patal_length = st.slider('patal_length', min_value=0.0, max_value=10.0, value=0.1)
    patal_width = st.slider('patal_width', min_value=0.0, max_value=10.0, value=0.1)
    
    if st.button('Make Prediction'):
        features = [sepal_length,sepal_width,patal_length,patal_width]
        result = make_prediction(features)
        st.success(f'The prediction is: {result}')

def make_prediction(features):
    # Use the loaded model to make predictions
    # Replace this with the actual code for your model
    input_array = np.array(features).reshape(1, -1)
    X_scaled = scaler.transform(input_array)
    prediction = model.predict(X_scaled)
    return prediction[0]

if __name__ == '__main__':
    main()

