import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from seaborn import load_dataset
from sklearn.model_selection import train_test_split


def load_model():
    if os.path.exists('model.pkl'):
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
    else:
        df = load_dataset("iris")
        le = LabelEncoder()
        df['species'] = le.fit_transform(df['species'])
        X = df.drop('species', axis=1)
        y = df['species']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        with open('model.pkl', 'wb') as f:
            pickle.dump(model, f)
    return model

def load_encoder():
    if os.path.exists('encoder.pkl'):
        with open('encoder.pkl', 'rb') as f:
            le = pickle.load(f)
    else:
        df = load_dataset("iris")
        le = LabelEncoder()
        le.fit_transform(df['species'])
        with open('encoder.pkl', 'wb') as f:
            pickle.dump(le, f)
    return le

st.set_page_config(page_title="Iris Classifier ", layout="centered")

st.title("Random Forest Iris Classifier")

model = load_model()
le = load_encoder()

st.subheader("Enter Iris Measurements")

col1, col2 = st.columns(2)

with col1:
    sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, max_value=10.0, value=5.0, step=0.1)
    petal_length = st.number_input("Petal Length (cm)", min_value=0.0, max_value=10.0, value=3.0, step=0.1)

with col2:
    sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, max_value=10.0, value=3.0, step=0.1)
    petal_width = st.number_input("Petal Width (cm)", min_value=0.0, max_value=3.0, value=1.0, step=0.1)

if st.button("Predict", type="primary"):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(input_data)[0]
    species_name = le.inverse_transform([prediction])[0]
    
    st.success(f"Predicted Species: **{species_name}**")
