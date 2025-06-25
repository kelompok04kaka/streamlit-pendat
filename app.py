import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler

@st.cache_data
def load_data():
    df = pd.read_csv("Sirosis (3).csv")
    return df

df = load_data()

selected_features = ['Bilirubin', 'Cholesterol', 'Albumin', 'Copper',
                     'Alk_Phos', 'SGOT', 'Tryglicerides', 'Platelets', 'Prothrombin']
X = df[selected_features]
y = df['Stage']

le_stage = LabelEncoder()
y_encoded = le_stage.fit_transform(y)
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)
model = SVC(kernel='linear', probability=True, random_state=42)
model.fit(X_scaled, y_encoded)

st.title('Prediksi Tahap Penyakit Sirosis')
st.subheader("Masukkan Nilai-nilai Parameter Klinis")

input_data = {}
for col in selected_features:
    default_val = float(df[col].mean())
    min_val = float(df[col].min())
    max_val = float(df[col].max())
    input_data[col] = st.number_input(f"{col}", min_value=min_val, max_value=max_val, value=default_val)

if st.button('Prediksi Stage'):
    input_df = pd.DataFrame([input_data])
    input_scaled = scaler.transform(input_df)
    
    pred_bin = model.predict(input_scaled)[0]
    pred_label = "Parah (D)" if pred_bin == 1 else "Tidak Parah (C/CL)"
    st.success(f"Status Sirosis: **{pred_label}**")