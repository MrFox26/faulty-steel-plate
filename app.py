import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("xgboost_model.pkl")

st.title("Faulty Steel Plate Prediction")
st.write("Enter the steel plate characteristics to predict the type of fault.")

# User Inputs
X_Minimum = st.number_input("X_Minimum", value=0.0)
X_Maximum = st.number_input("X_Maximum", value=0.0)
Y_Minimum = st.number_input("Y_Minimum", value=0.0)
Y_Maximum = st.number_input("Y_Maximum", value=0.0)
Pixels_Areas = st.number_input("Pixels_Areas", value=1.0)
X_Perimeter = st.number_input("X_Perimeter", value=1.0)
Y_Perimeter = st.number_input("Y_Perimeter", value=1.0)
Sum_of_Luminosity = st.number_input("Sum_of_Luminosity", value=1.0)
Minimum_of_Luminosity = st.number_input("Minimum_of_Luminosity", value=0.0)
Maximum_of_Luminosity = st.number_input("Maximum_of_Luminosity", value=0.0)
Length_of_Conveyer = st.number_input("Length_of_Conveyer", value=0.0)
TypeOfSteel = st.selectbox("TypeOfSteel", options=["A300", "A400"])
Steel_Plate_Thickness = st.number_input("Steel_Plate_Thickness", value=1.0)
Edges_Index = st.number_input("Edges_Index", value=0.0)
Empty_Index = st.number_input("Empty_Index", value=0.0)
Square_Index = st.number_input("Square_Index", value=0.0)
Outside_X_Index = st.number_input("Outside_X_Index", value=1.0)
Edges_X_Index = st.number_input("Edges_X_Index", value=0.0)
Edges_Y_Index = st.number_input("Edges_Y_Index", value=0.0)
Outside_Global_Index = st.number_input("Outside_Global_Index", value=1.0)

# Manual one-hot encoding for TypeOfSteel
A300 = 1 if TypeOfSteel == "A300" else 0
A400 = 1 if TypeOfSteel == "A400" else 0

# Build input DataFrame
input_data = pd.DataFrame([[
    X_Minimum, X_Maximum, Y_Minimum, Y_Maximum, Pixels_Areas,
    X_Perimeter, Y_Perimeter, Sum_of_Luminosity, Minimum_of_Luminosity,
    Maximum_of_Luminosity, Length_of_Conveyer, Steel_Plate_Thickness,
    Edges_Index, Empty_Index, Square_Index, Outside_X_Index,
    Edges_X_Index, Edges_Y_Index, Outside_Global_Index,
    A300, A400
]], columns=[
    'X_Minimum', 'X_Maximum', 'Y_Minimum', 'Y_Maximum', 'Pixels_Areas',
    'X_Perimeter', 'Y_Perimeter', 'Sum_of_Luminosity', 'Minimum_of_Luminosity',
    'Maximum_of_Luminosity', 'Length_of_Conveyer', 'Steel_Plate_Thickness',
    'Edges_Index', 'Empty_Index', 'Square_Index', 'Outside_X_Index',
    'Edges_X_Index', 'Edges_Y_Index', 'Outside_Global_Index',
    'A300', 'A400'
])

# Apply log1p transformations as in training
input_data['Pixels_Areas'] = np.log1p(input_data['Pixels_Areas'])
input_data['X_Perimeter'] = np.log1p(input_data['X_Perimeter'])
input_data['Y_Perimeter'] = np.log1p(input_data['Y_Perimeter'])
input_data['Sum_of_Luminosity'] = np.log1p(input_data['Sum_of_Luminosity'])
input_data['Outside_X_Index'] = np.log1p(input_data['Outside_X_Index'])
input_data['Outside_Global_Index'] = np.log1p(input_data['Outside_Global_Index'])
input_data['Steel_Plate_Thickness'] = np.log1p(input_data['Steel_Plate_Thickness'])

# Predict
if st.button("Predict Fault Type"):
    prediction = model.predict(input_data)[0]
    st.success(f"Predicted Fault Type (Label Encoded): {prediction}")
