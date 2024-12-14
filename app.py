import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import joblib

# Konfigurasi Streamlit
st.set_page_config(page_title="Customer Segmentation", layout="wide")
st.title("Customer Segmentation using K-Means Clustering")

# Load dataset dari file lokal
df = pd.read_csv("Customer Purchase Data.csv")  # Ganti dengan nama file dataset lokal Anda
st.write("### Dataset")
st.write(df.head())

# Menampilkan missing values
st.write("### Missing Values")
st.write(df.isnull().sum())

# Pilih variabel untuk clustering
variables = st.multiselect(
    "Pilih Variabel untuk Clustering:",
    df.columns,
    default=['Income', 'Spending_Score', 'Last_Purchase_Amount', 'Membership_Years']
)

if len(variables) >= 2:
    # Visualisasi distribusi setiap variabel
    st.write("### Distribusi Variabel")
    for var in variables:
        st.write(f"Distribusi {var}")
        fig, ax = plt.subplots()
        sns.histplot(df[var], kde=True, bins=30, ax=ax)
        st.pyplot(fig)

    # Scatter plot antar variabel
    st.write("### Pairplot Antar Variabel")
    pairplot_fig = sns.pairplot(df[variables])
    st.pyplot(pairplot_fig)

    # Normalisasi data
    st.write("### Normalisasi Data")
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df[variables])
    df_scaled = pd.DataFrame(df_scaled, columns=variables)
    st.write(df_scaled.head())

    # Elbow Method untuk menentukan jumlah cluster
    st.write("### Elbow Method")
    inertia = []
    k_range = range(1, 11)
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(df_scaled)
        inertia.append(kmeans.inertia_)

    fig, ax = plt.subplots()
    ax.plot(k_range, inertia, 'bo-')
    ax.set_title('Elbow Method untuk Menentukan Jumlah Cluster')
    ax.set_xlabel('Jumlah Cluster (k)')
    ax.set_ylabel('Inertia')
    st.pyplot(fig)

    # Silhouette Score untuk menentukan jumlah cluster
    st.write("### Silhouette Score")
    silhouette_scores = []
    for k in range(2, 11):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(df_scaled)
        score = silhouette_score(df_scaled, kmeans.labels_)
        silhouette_scores.append(score)

    fig, ax = plt.subplots()
    ax.plot(range(2, 11), silhouette_scores, 'bo-')
    ax.set_title('Silhouette Score untuk Menentukan Jumlah Cluster')
    ax.set_xlabel('Jumlah Cluster (k)')
    ax.set_ylabel('Silhouette Score')
    st.pyplot(fig)

    # Input jumlah cluster optimal dari pengguna
    optimal_k = st.number_input("Pilih Jumlah Cluster (k):", min_value=2, max_value=10, value=3, step=1)

    # K-Means Clustering
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    df['cluster'] = kmeans.fit_predict(df_scaled)

    # Visualisasi hasil clustering
    st.write("### Hasil Clustering")
    pairplot_fig = sns.pairplot(df, hue='cluster', vars=variables, palette='Set1')
    st.pyplot(pairplot_fig)

    # Menampilkan jumlah data pada tiap cluster
    st.write("### Jumlah Data pada Tiap Cluster")
    st.write(df['cluster'].value_counts())

    # Analisis statistik tiap cluster
    st.write("### Statistik Tiap Cluster")
    st.write(df.groupby('cluster')[variables].mean())

    # Save Model
    if st.button("Simpan Model K-Means"):
        joblib.dump(kmeans, 'kmeans_model.pkl')
        st.success("Model berhasil disimpan sebagai 'kmeans_model.pkl'")

else:
    st.warning("Pilih minimal 2 variabel untuk clustering.")

# Prediksi Data Baru Menggunakan Model
st.write("## Prediksi Cluster dengan Model K-Means")
model_file = st.file_uploader("Upload Model K-Means (PKL)", type=["pkl"])
if model_file is not None:
    model = joblib.load(model_file)
    st.write("Model berhasil diunggah!")

    # Pastikan variabel yang digunakan sama seperti saat melatih model
    with st.form("prediction_form"):
        st.write("Masukkan Data Baru untuk Prediksi Cluster")
        input_data = []

        # Hanya gunakan variabel yang sama seperti saat pelatihan
        for var in variables:
            value = st.slider(f"{var}", min_value=float(df[var].min()), max_value=float(df[var].max()), step=0.1, value=float(df[var].mean()))
            input_data.append(value)

        submit_prediction = st.form_submit_button("Predict")

        if submit_prediction:
            new_data = np.array([input_data])
            if new_data.shape[1] == model.n_features_in_:
                predicted_cluster = model.predict(new_data)
                st.subheader("Hasil Prediksi:")
                st.write(f"Data baru masuk ke cluster: *{predicted_cluster[0]}*")
            else:
                st.error(f"Jumlah fitur tidak sesuai. Model ini mengharapkan {model.n_features_in_} fitur, tetapi input memiliki {new_data.shape[1]} fitur.")

# Analisis statistik tiap cluster untuk interpretasi
st.write("### Statistik Tiap Cluster")
cluster_analysis = df.groupby('cluster')[variables].mean()
st.write(cluster_analysis)

# Interpretasi hasil clustering
st.write("### Interpretasi Cluster")
# Misalnya menggunakan Spending_Score untuk label
cluster_labels = {}
for cluster, data in cluster_analysis.iterrows():
    if data['Spending_Score'] < 0.33 * df['Spending_Score'].max():
        cluster_labels[cluster] = "Hemat"
    elif data['Spending_Score'] < 0.66 * df['Spending_Score'].max():
        cluster_labels[cluster] = "Sedang"
    else:
        cluster_labels[cluster] = "Boros"

# Menambahkan interpretasi cluster ke dataframe
df['cluster_label'] = df['cluster'].map(cluster_labels)
st.write(df[['cluster', 'cluster_label']].drop_duplicates().sort_values('cluster'))

# Visualisasi hasil clustering dengan label interpretasi
st.write("### Visualisasi Hasil Clustering dengan Label Interpretasi")
pairplot_fig = sns.pairplot(df, hue='cluster_label', vars=variables, palette='Set2')
st.pyplot(pairplot_fig)

if submit_prediction:
    new_data = np.array([input_data])
    if new_data.shape[1] == model.n_features_in_:
        predicted_cluster = model.predict(new_data)
        predicted_label = cluster_labels.get(predicted_cluster[0], "Unknown")
        st.subheader("Hasil Prediksi:")
        st.write(f"Data baru masuk ke cluster: *{predicted_cluster[0]}* dengan label: *{predicted_label}*")
    else:
        st.error(f"Jumlah fitur tidak sesuai. Model ini mengharapkan {model.n_features_in_} fitur, tetapi input memiliki {new_data.shape[1]} fitur.")
