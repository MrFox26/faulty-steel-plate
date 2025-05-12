# app.py

import streamlit as st
import numpy as np
import pandas as pd
import joblib
from xgboost import XGBClassifier

# Load the trained model
model = joblib.load('xgboost_model.pkl')

# Define the label mapping used earlier
label_mapping = {
    'Bumps': 0,
    'Dirtiness': 1,
    'K_Scatch': 2,
    'Other_Faults': 3,
    'Pastry': 4,
    'Stains': 5,
    'Z_Scratch': 6
}
inv_label_mapping = {v: k for k, v in label_mapping.items()}

# UI title
st.title("Steel Fault Type Prediction App")

st.write("Enter the features to predict the fault type.")

# Feature input
features = {}

numerical_columns = [
    'X_Minimum', 'X_Maximum', 'Y_Minimum', 'Y_Maximum', 'Pixels_Areas',
    'X_Perimeter', 'Y_Perimeter', 'Sum_of_Luminosity', 'Minimum_of_Luminosity',
    'Maximum_of_Luminosity', 'Length_of_Conveyer', 'TypeOfSteel', 
    'Outside_X_Index', 'Steel_Plate_Thickness', 'Edges_Index', 'Empty_Index', 
    'Square_Index', 'Outside_Global_Index'
]

# Create inputs for all features
for col in numerical_columns:
    features[col] = st.number_input(f"Enter {col}", value=0.0)

# Steel_type input (binary: 0 or 1 after LabelEncoding)
steel_type = st.selectbox("Steel Type", options=['A300', 'A400'])
features['Steel_type'] = 0 if steel_type == 'A300' else 1

# Predict button
if st.button("Predict Fault Type"):
    input_df = pd.DataFrame([features])
    prediction = model.predict(input_df)[0]
    st.success(f"Predicted Fault Type: **{inv_label_mapping[prediction]}**")
