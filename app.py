import streamlit as st
import pandas as pd
import joblib

# Load model terlatih
model = joblib.load('decision_tree_model.pkl')  # Ubah jika model yang dipakai berbeda

# Judul aplikasi
st.title('Prediksi Risiko Kesehatan Ibu Hamil')
st.subheader('Berdasarkan Faktor Fisiologis')

st.markdown("""
Aplikasi ini menggunakan model machine learning untuk memprediksi tingkat risiko kesehatan ibu 
berdasarkan beberapa indikator medis seperti usia, tekanan darah, gula darah, dan lainnya.
""")

# Input data pengguna
age = st.slider('Usia (tahun)', 15, 50, 30)
systolic_bp = st.slider('Tekanan Darah Sistolik (mmHg)', 90, 200, 120)
diastolic_bp = st.slider('Tekanan Darah Diastolik (mmHg)', 50, 130, 80)
bs = st.slider('Gula Darah (mmol/L)', 3.0, 20.0, 6.0)
body_temp = st.slider('Suhu Tubuh (°C)', 35.0, 41.0, 37.0)
heart_rate = st.slider('Denyut Jantung (BPM)', 50, 180, 80)

# Tombol prediksi
if st.button('Prediksi Risiko'):
    # Bentuk data untuk model
    input_df = pd.DataFrame([[age, systolic_bp, diastolic_bp, bs, body_temp, heart_rate]],
                            columns=['Age', 'SystolicBP', 'DiastolicBP', 'BS', 'BodyTemp', 'HeartRate'])
    
    # Prediksi
    prediction = model.predict(input_df)[0]
    
    # Tampilkan hasil prediksi
    st.markdown("### Hasil Prediksi:")
    if prediction == 'low':
        st.success('✅ Risiko Kesehatan Ibu: RENDAH')
    elif prediction == 'mid':
        st.warning('⚠️ Risiko Kesehatan Ibu: SEDANG')
    else:
        st.error('❌ Risiko Kesehatan Ibu: TINGGI')
