# create environment for windows
# python -m venv myenv
# activate environment
# myenv\Scripts\activate
# pip install streamlit scikit-learn pandas numpy

import streamlit as st
import pandas as pd
import pickle
import os

# -------------------------------
# Load model and scaler
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = pickle.load(open(os.path.join(BASE_DIR, "model.pkl"), "rb"))
scaler = pickle.load(open(os.path.join(BASE_DIR, "scaler.pkl"), "rb"))

# -------------------------------
# UI
# -------------------------------
st.set_page_config(page_title="House Price Prediction", page_icon="🏠")

st.title("🏠 House Price Prediction App")
st.write("Enter the details below to predict house price")

# -------------------------------
# Inputs
# -------------------------------
col1, col2 = st.columns(2)

with col1:
    sqft = st.number_input('Square Footage', 500, 5000, 2000)
    bed = st.number_input('Bedrooms', 1, 5, 3)
    bath = st.number_input('Bathrooms', 1, 3, 2)
    year = st.number_input('Year Built', 1900, 2025, 2000)

with col2:
    lot = st.number_input('Lot Size', 100, 10000, 2000)
    garage = st.number_input('Garage Size', 0, 3, 1)
    neigh = st.number_input('Neighborhood Quality', 1, 10, 5)

# -------------------------------
# DataFrame
# -------------------------------
input_data = pd.DataFrame({
    'Square_Footage': [sqft],
    'Num_Bedrooms': [bed],
    'Num_Bathrooms': [bath],
    'Year_Built': [year],
    'Lot_Size': [lot],
    'Garage_Size': [garage],
    'Neighborhood_Quality': [neigh]
})

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict Price"):

    # scale input
    input_scaled = scaler.transform(input_data)

    # keep column names (removes warning)
    input_scaled = pd.DataFrame(input_scaled, columns=input_data.columns)

    # predict
    prediction = model.predict(input_scaled)
    prediction = prediction / 100   # adjust scale

    # display
    st.success(f"💰 Estimated House Price: ${prediction[0]:,.2f}")
