import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os

BASE_DIR = os.path.dirname(__file__)
scaler = joblib.load(os.path.join(BASE_DIR, "preprocessor.pkl"))
model = joblib.load(os.path.join(BASE_DIR, "model.pkl"))

def main():
    st.title('Machine Learning Heart Attack Prediction Model Deployment')

    gender_options = {"Pria": 1, "Wanita": 0}
    gender = st.selectbox("Pilih Gender", options=list(gender_options.keys()))