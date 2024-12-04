import streamlit as st
import joblib
import numpy as np

# Load model K-Means
model = joblib.load('kmeans_model.pkl')

# Judul aplikasi
st.title("Prediksi Cluster dengan K-Means")
st.write("Gunakan slider di bawah untuk memasukkan data baru dan prediksi cluster.")

with st.form('prediction'):
  sepal_length = st.slider("Sepal Length (cm)", min_value=4.0, max_value=8.0, step=0.1, value=5.0)
  sepal_width = st.slider("Sepal Width (cm)", min_value=2.0, max_value=4.5, step=0.1, value=3.5)
  petal_length = st.slider("Petal Length (cm)", min_value=1.0, max_value=7.0, step=0.1, value=1.4)
  petal_width = st.slider("Petal Width (cm)", min_value=0.1, max_value=2.5, step=0.1, value=0.2)

  submit_button = st.form_submit_button(label='Predict')

if submit_button:
  new_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
  predicted_cluster = model.predict(new_data)
  st.subheader("Hasil Prediksi:")
  st.write(f"Data baru masuk ke cluster: *{predicted_cluster[0]}*")