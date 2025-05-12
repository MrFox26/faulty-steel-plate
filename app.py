import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

# Streamlit App
st.title("Steel Plate Fault Prediction")

uploaded_file = st.file_uploader("Upload faults.csv", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Raw Data Sample", df.sample(5))

    # Combine multiple target columns into one
    target_cols = df.columns[-7:]
    df['target'] = df[target_cols].idxmax(axis=1)
    df = df.drop(columns=target_cols)

    X = df.drop(columns=['target'])
    y = df['target']

    st.write("### Class Distribution", y.value_counts())

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    model_choice = st.selectbox("Choose Model", ["Random Forest", "XGBoost", "Decision Tree", "SVM"])

    if st.button("Train Model"):
        if model_choice == "Random Forest":
            model = RandomForestClassifier(random_state=42)
        elif model_choice == "XGBoost":
            model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
        elif model_choice == "Decision Tree":
            model = DecisionTreeClassifier(random_state=42)
        elif model_choice == "SVM":
            model = SVC(probability=True)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        st.write("### Classification Report")
        st.text(classification_report(y_test, y_pred))

        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        st.pyplot(fig)

        st.success("Model trained and evaluated!")
