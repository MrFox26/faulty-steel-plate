# app.py
import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the trained model
model = joblib.load("xgboost_model.pkl")

# Page title
st.title("Faulty Steel Plate Prediction App")

st.write("Enter the features below to predict the type of fault:")

# User inputs
X_Minimum = st.number_input("X_Minimum")
X_Maximum = st.number_input("X_Maximum")
Y_Minimum = st.number_input("Y_Minimum")
Y_Maximum = st.number_input("Y_Maximum")
Pixels_Areas = st.number_input("Pixels_Areas")
X_Perimeter = st.number_input("X_Perimeter")
Y_Perimeter = st.number_input("Y_Perimeter")
Sum_of_Luminosity = st.number_input("Sum_of_Luminosity")
Minimum_of_Luminosity = st.number_input("Minimum_of_Luminosity")
Maximum_of_Luminosity = st.number_input("Maximum_of_Luminosity")
Length_of_Conveyer = st.number_input("Length_of_Conveyer")
TypeOfSteel = st.selectbox("Type Of Steel", ["A300", "A400"])
Steel_Plate_Thickness = st.number_input("Steel_Plate_Thickness")
Edges_Index = st.number_input("Edges_Index")
Empty_Index = st.number_input("Empty_Index")
Square_Index = st.number_input("Square_Index")
Outside_X_Index = st.number_input("Outside_X_Index")
Edges_X_Index = st.number_input("Edges_X_Index")
Edges_Y_Index = st.number_input("Edges_Y_Index")
Outside_Global_Index = st.number_input("Outside_Global_Index")

# One-hot encoding manually
A300 = 1 if TypeOfSteel == "A300" else 0
A400 = 1 if TypeOfSteel == "A400" else 0

# Combine all features into a DataFrame
input_data = pd.DataFrame([[
    X_Minimum, X_Maximum, Y_Minimum, Y_Maximum,
    Pixels_Areas, X_Perimeter, Y_Perimeter,
    Sum_of_Luminosity, Minimum_of_Luminosity, Maximum_of_Luminosity,
    Length_of_Conveyer, Steel_Plate_Thickness,
    Edges_Index, Empty_Index, Square_Index, Outside_X_Index,
    Edges_X_Index, Edges_Y_Index, Outside_Global_Index,
    A300, A400
]], columns=[
    'X_Minimum', 'X_Maximum', 'Y_Minimum', 'Y_Maximum',
    'Pixels_Areas', 'X_Perimeter', 'Y_Perimeter',
    'Sum_of_Luminosity', 'Minimum_of_Luminosity', 'Maximum_of_Luminosity',
    'Length_of_Conveyer', 'Steel_Plate_Thickness',
    'Edges_Index', 'Empty_Index', 'Square_Index', 'Outside_X_Index',
    'Edges_X_Index', 'Edges_Y_Index', 'Outside_Global_Index',
    'A300', 'A400'
])

# Predict
if st.button("Predict Fault Type"):
    prediction = model.predict(input_data)[0]
    st.success(f"Predicted Fault Type: {prediction}")
