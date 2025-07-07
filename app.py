import streamlit as st
import numpy as np
import pickle

# Load models
dtr = pickle.load(open('models/dtr.pkl', 'rb'))
preprocessor = pickle.load(open('models/preprocessor.pkl', 'rb'))

st.title("Crop Yield Prediction")

with st.form("prediction_form"):
    Year = st.text_input("Year")
    average_rain_fall_mm_per_year = st.text_input("Average Rainfall (mm/year)")
    pesticides_tonnes = st.text_input("Pesticides Used (tonnes)")
    avg_temp = st.text_input("Average Temperature")
    Area = st.text_input("Area")
    Item = st.text_input("Crop Type (Item)")

    submit = st.form_submit_button("Predict")

if submit:
    features = np.array([[Year, average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp, Area, Item]], dtype=object)
    transformed_features = preprocessor.transform(features)
    prediction = dtr.predict(transformed_features)

    st.success(f"🌾 Predicted Crop Yield: {prediction[0]:.2f}")
