import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load model
model = joblib.load("xgboost_model.pkl")

# Target label mapping
target_mapping = {0: 'Bumps', 1: 'Dirtiness', 2: 'K_Scatch', 3: 'Other_Faults', 4: 'Pastry', 5: 'Stains', 6: 'Z_Scratch'}

# Optional custom CSS
st.markdown("""
<style>
    .stButton>button {
        color: white;
        background-color: #1f77b4;
        border-radius: 5px;
        height: 3em;
        width: 100%;
    }
    .stMetric {
        font-size: 20px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("🧾 About")
    st.markdown("""
    This app predicts **fault types** in steel plates using a pre-trained XGBoost model.
    
    **Features**:
    - 27 inputs
    - Real-time prediction
    - Label-decoded result
    """)

# Title and description
st.title("🔍 Faulty Steel Plate Prediction")
st.markdown("""
Enter the steel plate characteristics below to predict the **type of fault**.
""")

# Basic Features
with st.expander("📐 Basic Dimensions"):
    col1, col2 = st.columns(2)
    with col1:
        X_Minimum = st.number_input("X_Minimum", value=42.0)
        X_Maximum = st.number_input("X_Maximum", value=50.0)
        Y_Minimum = st.number_input("Y_Minimum", value=270900.0)
        Y_Maximum = st.number_input("Y_Maximum", value=270944.0)
    with col2:
        Pixels_Areas = st.number_input("Pixels_Areas", value=267.0)
        X_Perimeter = st.number_input("X_Perimeter", value=17.0)
        Y_Perimeter = st.number_input("Y_Perimeter", value=44.0)

# Luminosity
with st.expander("💡 Luminosity Features"):
    col3, col4 = st.columns(2)
    with col3:
        Sum_of_Luminosity = st.number_input("Sum_of_Luminosity", value=24220.0)
        Minimum_of_Luminosity = st.number_input("Minimum_of_Luminosity", value=76.0)
        Maximum_of_Luminosity = st.number_input("Maximum_of_Luminosity", value=108.0)
    with col4:
        Luminosity_Index = st.number_input("Luminosity_Index", value=-0.2913)

# Other features
with st.expander("⚙️ Advanced Features"):
    col5, col6 = st.columns(2)
    with col5:
        Length_of_Conveyer = st.number_input("Length_of_Conveyer", value=1687.0)
        Steel_Plate_Thickness = st.number_input("Steel_Plate_Thickness", value=80.0)
        Edges_Index = st.number_input("Edges_Index", value=0.0498)
        Empty_Index = st.number_input("Empty_Index", value=0.2415)
        Square_Index = st.number_input("Square_Index", value=0.1818)
        Outside_X_Index = st.number_input("Outside_X_Index", value=0.0047)
    with col6:
        Edges_X_Index = st.number_input("Edges_X_Index", value=0.4706)
        Edges_Y_Index = st.number_input("Edges_Y_Index", value=1)
        Outside_Global_Index = st.selectbox("Outside_Global_Index", options=[1, 0.5, 0])
        LogOfAreas = st.number_input("LogOfAreas", value=2.4265)
        Log_X_Index = st.number_input("Log_X_Index", value=0.9031)
        Log_Y_Index = st.number_input("Log_Y_Index", value=1.6435)
        Orientation_Index = st.number_input("Orientation_Index", value=0.8182)
        SigmoidOfAreas = st.number_input("SigmoidOfAreas", value=0.5822)

# Steel type
Steel_type = st.selectbox("🛠️ Steel Type", options=["A300", "A400"])
Steel_type_encoded = 0 if Steel_type == "A300" else 1

# Prepare input
input_data = pd.DataFrame([[
    X_Minimum, X_Maximum, Y_Minimum, Y_Maximum, Pixels_Areas,
    X_Perimeter, Y_Perimeter, Sum_of_Luminosity, Minimum_of_Luminosity,
    Maximum_of_Luminosity, Length_of_Conveyer, Steel_Plate_Thickness,
    Edges_Index, Empty_Index, Square_Index, Outside_X_Index,
    Edges_X_Index, Edges_Y_Index, Outside_Global_Index, LogOfAreas,
    Log_X_Index, Log_Y_Index, Orientation_Index, Luminosity_Index,
    SigmoidOfAreas, Steel_type_encoded
]], columns=[
    'X_Minimum', 'X_Maximum', 'Y_Minimum', 'Y_Maximum', 'Pixels_Areas',
    'X_Perimeter', 'Y_Perimeter', 'Sum_of_Luminosity', 'Minimum_of_Luminosity',
    'Maximum_of_Luminosity', 'Length_of_Conveyer', 'Steel_Plate_Thickness',
    'Edges_Index', 'Empty_Index', 'Square_Index', 'Outside_X_Index',
    'Edges_X_Index', 'Edges_Y_Index', 'Outside_Global_Index', 'LogOfAreas',
    'Log_X_Index', 'Log_Y_Index', 'Orientation_Index', 'Luminosity_Index',
    'SigmoidOfAreas', 'Steel_type'
])

# Apply transformations
input_data['Pixels_Areas'] = np.log1p(input_data['Pixels_Areas'])
input_data['X_Perimeter'] = np.log1p(input_data['X_Perimeter'])
input_data['Y_Perimeter'] = np.log1p(input_data['Y_Perimeter'])
input_data['Sum_of_Luminosity'] = np.log1p(input_data['Sum_of_Luminosity'])
input_data['Outside_X_Index'] = np.log1p(input_data['Outside_X_Index'])
input_data['Outside_Global_Index'] = np.log1p(input_data['Outside_Global_Index'])
input_data['Steel_Plate_Thickness'] = np.log1p(input_data['Steel_Plate_Thickness'])

# Display input summary
st.subheader("🧾 Input Summary")
st.dataframe(input_data)

# Prediction
if st.button("🔍 Predict Fault Type"):
    prediction = model.predict(input_data)[0]
    predicted_label = target_mapping.get(prediction, "Unknown")
    st.metric("Predicted Fault Type", f"{predicted_label}")
