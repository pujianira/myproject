import streamlit as st
st.markdown("""
# **Customer Behavior Segmentation Analysis** 

# 👯‍♀️ **Anggota Kelompok:** 👯‍♀️
1️⃣ Pujiani Rahayu Agustin - 24060122130067

2️⃣ Meyta Rizki Khairunisa - 24060122130085

3️⃣ Aura Arfannisa Az Zahra - 24060122130097

4️⃣ Nabila Betari Anjani - 24060122140169

# ‼️**Identifikasi Masalah**⁉️
Sebuah perusahaan retail menyadari bahwa pelanggan mereka memiliki kebiasaan belanja yang beragam, sehingga perlu membuat strategi pemasaran yang lebih spesifik menyasar pelanggan dengan masing-masing kebiasan mereka. Perusahaan ingin membagi pelanggannya menjadi tiga kategori, yaitu **pelanggan hemat, pelanggan biasa, dan pelanggan dengan pengeluaran tinggi.** Masing-masing kategori nantinya akan diberikan strategi pemasaran yang sesuai.

Dengan menggunakan dataset yang ada, kami menganalisis data dari berbagai kriteria, seperti usia dan pendapatan pelanggan, skor pengeluaran mereka, keanggotaan mereka, frekuensi pembelian dan jumlah pembelian terakhir mereka untuk menentukan clustering menjadi tiga kategori pelanggan.

# 📚 **Library yang Diperlukan**
""")

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.tokenize import word_tokenize
from sklearn.metrics import silhouette_score
import re
import io

"""#  📂**1.Import & Load Dataset**🧩

**a. Import dataset**
"""

# from google.colab import files
# uploaded = files.upload()

df = pd.read_csv('Customer Purchase Data.csv')

"""**b. Lima baris awal dari dataset**"""
st.write(df.head())
"""**c. Eksplorasi baris & kolom, tipe data**"""

st.subheader("1.2 Data Information")
buffer = io.StringIO()
df.info(buf=buffer)
st.text(buffer.getvalue())

"""**d. Eksplorasi missing values**"""

st.write(df.isnull().sum())

"""**e. Eksplorasi data duplikat**"""

st.write((f"Jumlah duplikasi: {df.duplicated().sum()}"))
df = df.drop_duplicates()

"""**f. Statistik deskriptif dataset**"""

st.write(df.describe())

"""**g. Penghapusan kolom 'Number'**"""

#Drop kolom number
df.drop(["Number"], axis=1, inplace=True)

st.write(df.columns)

"""# **2. Visualisasi Distribusi Variabel Numerik**

**a. Visualisasi distribusi data menggunakan histogram**
"""

import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler

# === Data Cleaning ===
# Menghapus nilai null
df_cleaned = df.dropna()

# Drop kolom 'Number' jika ada
if 'Number' in df_cleaned.columns:
    df_cleaned.drop(["Number"], axis=1, inplace=True)

# Menghapus duplikasi
df_cleaned = df_cleaned.drop_duplicates()

# Kolom numerik untuk analisis
numerical_columns = ['Age', 'Income', 'Spending_Score', 'Membership_Years', 'Purchase_Frequency', 'Last_Purchase_Amount']
# Daftar variabel yang ingin diplot
variables = ['Age', 'Income', 'Spending_Score', 'Membership_Years', 'Purchase_Frequency', 'Last_Purchase_Amount']

# **a. Visualisasi distribusi data menggunakan histogram**
st.subheader("a. Visualisasi distribusi data menggunakan histogram")

# Cek apakah kolom-kolom ada di dalam DataFrame
for var in variables:
    if var in df.columns:
        st.write(f"Menampilkan distribusi untuk kolom: {var}")

        # Cek tipe data kolom
        if pd.api.types.is_numeric_dtype(df[var]):
            # Menghapus NaN sebelum plotting
            df[var].dropna(inplace=True)

            # Membuat plot histogram dengan KDE
            fig, ax = plt.subplots()
            sns.histplot(df[var], kde=True, bins=30, ax=ax)
            ax.set_title(f'Distribusi {var}')
            st.pyplot(fig)
        else:
            st.write(f"Kolom {var} bukan tipe numerik.")
    else:
        st.write(f"Kolom {var} tidak ditemukan dalam dataset.")

# **b. Visualisasi distribusi outlier menggunakan boxplot**
st.subheader("b. Visualisasi distribusi outlier menggunakan boxplot")

# Boxplot untuk melihat distribusi dan outlier
fig, ax = plt.subplots(figsize=(15, 20))
df.plot(kind='box',
        subplots=True,
        layout=(6, 3),  # Ubah layout agar lebih besar dari jumlah fitur
        sharex=False,
        sharey=False,
        ax=ax)  # Ukuran gambar lebih besar untuk tampilan jelas
plt.tight_layout()
st.pyplot(fig)

# **c. Visualisasi matriks korelasi menggunakan heatmap**
st.subheader("c. Visualisasi matriks korelasi menggunakan heatmap")

# Scaling data (pastikan semua kolom ada)
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df_cleaned[numerical_columns]), columns=numerical_columns)

# Matriks korelasi
correlation_matrix = df_scaled.corr()

# Menampilkan matriks korelasi dengan lebih banyak desimal
st.write("=== Matriks Korelasi ===")
st.write(correlation_matrix.round(4))  # Membulatkan ke 4 desimal

# Visualisasi dengan lebih banyak desimal
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.4f', linewidths=0.5, ax=ax)  # Format 4 desimal
ax.set_title('Matriks Korelasi antar Variabel')
st.pyplot(fig)

# Memeriksa kolom yang hilang
missing_columns = [col for col in numerical_columns if col not in df_scaled.columns]
st.write("Kolom yang hilang:", missing_columns)


"""# **3. Analisis Korelasi Antarfitur dalam Dataset**

Berdasarkan matriks korelasi pada diaggram heatmap di atas, dapat ditarik kesimpulan bahwa:
"""

import pandas as pd

# Data hubungan korelasi antar fitur
data = {
    'Fitur': ['Income', 'Age', 'Spending_Score', 'Purchase_Frequency', 'Membership_Years', 'Last_Purchase_Amount'],
    'Berkorelasi Dengan': [
        'Age, Last_Purchase_Amount, Spending_Score',
        'Income, Last_Purchase_Amount, Spending_Score',
        'Age, Income, Last_Purchase_Amount',
        'Membership_Years',
        'Purchase_Frequency',
        'Age, Income, Spending_Score'
    ]
}

# Membuat DataFrame dari data
correlation_table = pd.DataFrame(data)

# Menampilkan tabel korelasi
correlation_table

"""Jika dibagi menjadi 2 cluster, maka hasil dari korelasi antarfitur tersebut adalah:"""

import pandas as pd

# Data kelompok dan fitur yang berkorelasi
data = {
    'Principal Component': ['PC1 (Variasi Ekonomi)', 'PC2 (Dimensi Waktu Membership)'],
    'Fitur': [
        'Income, Age, Last_Purchase_Amount, Spending_Score',
        'Membership_Years, Purchase_Frequency'
    ],
    'Keterangan': [
        'Pelanggan dengan pendapatan lebih tinggi mungkin juga memiliki skor belanja yang lebih tinggi atau melakukan pembelian lebih sering',
        'Pelanggan yang lebih lama menjadi anggota mungkin lebih sering melakukan pembelian'
    ]
}

# Membuat DataFrame dari data
group_table = pd.DataFrame(data)

# Menampilkan tabel kelompok
group_table

"""# 🎲**4. Dimensionality Reduction**

**a. Outlier handling**
"""
# 2. Outlier Detection dan Handling menggunakan IQR method
def handle_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Replace outliers with bounds
    df[column] = np.where(df[column] > upper_bound, upper_bound,
                         np.where(df[column] < lower_bound, lower_bound, df[column]))
    return df

# Columns untuk outlier handling
numerical_columns = ['Age', 'Income', 'Spending_Score', 'Membership_Years',
                    'Purchase_Frequency', 'Last_Purchase_Amount']

# Outlier handling untuk setiap kolom numerik
for column in numerical_columns:
    df = handle_outliers(df, column)

# Visualisasi boxplot setelah handling outlier
st.subheader('Visualisasi Boxplot setelah Penanganan Outlier')

fig, ax = plt.subplots(figsize=(15, 20))
df[numerical_columns].plot(kind='box', subplots=True, layout=(6, 3), 
                            sharex=False, sharey=False, figsize=(15, 20), ax=ax)
plt.tight_layout()

# Menampilkan plot di Streamlit
st.pyplot(fig)

"""# 🧩**5. Principal Component Analysis (PCA)**🤖"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Fungsi untuk menganalisis korelasi antar fitur
def analyze_correlations(correlation_matrix, threshold=0.7):
    # Mencari pasangan fitur dengan korelasi tinggi
    high_correlations = []

    # Iterasi untuk setiap pasangan fitur
    for i in range(len(correlation_matrix.columns)):
        for j in range(i + 1, len(correlation_matrix.columns)):
            correlation = abs(correlation_matrix.iloc[i, j])
            if correlation > threshold:
                high_correlations.append({
                    'Feature 1': correlation_matrix.columns[i],
                    'Feature 2': correlation_matrix.columns[j],
                    'Correlation': correlation
                })

    # Mengubah hasil menjadi DataFrame dan mengurutkan berdasarkan nilai korelasi
    results_df = pd.DataFrame(high_correlations)
    if not results_df.empty:
        results_df = results_df.sort_values('Correlation', ascending=False)

    return results_df

# Skalakan data
scaler = StandardScaler()
df_scaled = df_cleaned.copy()
df_scaled[numerical_columns] = scaler.fit_transform(df_cleaned[numerical_columns])

# Matriks korelasi antar fitur
correlation_matrix = df_scaled[numerical_columns].corr()

# Menganalisis korelasi dengan threshold 0.7
correlations = analyze_correlations(correlation_matrix, threshold=0.7)

print("\nPasangan fitur dengan korelasi tinggi:")
print(correlations.to_string(index=False))

# Mengidentifikasi kelompok fitur yang berkorelasi
print("\nKelompok fitur yang saling berkorelasi:")

# Menentukan ambang batas korelasi
threshold = 0.9

# Dictionary untuk menyimpan hubungan korelasi
correlation_relationships = {}

# Identifikasi fitur redundan dan hubungan korelasi antar fitur
for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > threshold:
            colname_1 = correlation_matrix.columns[i]
            colname_2 = correlation_matrix.columns[j]

            # Menyimpan fitur yang berkorelasi dalam bentuk set untuk menghindari duplikasi
            if colname_1 not in correlation_relationships:
                correlation_relationships[colname_1] = set()
            correlation_relationships[colname_1].add(colname_2)

            if colname_2 not in correlation_relationships:
                correlation_relationships[colname_2] = set()
            correlation_relationships[colname_2].add(colname_1)

# Menampilkan hasil hubungan korelasi
for feature, correlated_features in correlation_relationships.items():
    correlated_features_list = ', '.join(sorted(correlated_features))
    print(f"{feature}\t->\t{correlated_features_list}")

# PCA
pca = PCA(n_components=2)
principal_components = pca.fit_transform(df_scaled[numerical_columns])

# Buat DataFrame baru dengan hasil PCA
pca_data = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])

# Buat DataFrame komponen utama
components_df = pd.DataFrame(pca.components_, columns=df_scaled[numerical_columns].columns, index=['PC1', 'PC2'])

# Menampilkan nama kolom penyusun setiap komponen utama
print("\nKomponen Penyusun untuk PC1 dan PC2:")
for i, row in components_df.iterrows():
    print(f"Komponen penyusun {i}:")
    # Filter fitur dengan kontribusi signifikan (threshold 0.2)
    contributing_features = row.index[row.abs() > 0.2]
    # Menampilkan nama kolom
    print(", ".join(contributing_features))  # Gabungkan nama kolom dengan koma
    print()

"""**Analisis:**

**Komponen penyusun PC1: Age, Income, Spending_Score, Last_Purchase_Amount**

1. Kelompok fitur ini saling berkorelasi satu sama lain. Misalnya, Income berhubungan erat dengan Age, Last_Purchase_Amount, dan Spending_Score

2. Menunjukkan bahwa pelanggan dengan pendapatan lebih tinggi mungkin juga memiliki skor belanja yang lebih tinggi atau melakukan pembelian lebih sering.

**Komponen penyusun PC2: Membership_Years, Purchase_Frequency**

1. Fitur dalam kelompok ini memiliki korelasi yang jelas satu sama lain, di mana Membership_Years berhubungan dengan Purchase_Frequency.
2. Menunjukkan bahwa pelanggan yang lebih lama menjadi anggota mungkin lebih sering melakukan pembelian

# 🎮 **6. Penentuan K dengan Elbow Method & Silhouette Score**🎯

**a. Elbow Method**
"""
# Mencari jumlah cluster optimal dengan Elbow Method
inertia = []
k_values = range(2, 10)

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(df_scaled)
    inertia.append(kmeans.inertia_)

# Visualisasi Elbow Curve
st.subheader('Elbow Method untuk Menentukan Jumlah Cluster Optimal')

fig, ax = plt.subplots(figsize=(10, 6))
sns.lineplot(x=k_values, y=inertia, marker='o', ax=ax)
ax.set_title('Elbow Method for Optimal k')
ax.set_xlabel('Number of Clusters (k)')
ax.set_ylabel('Inertia')
ax.set_xticks(k_values)
ax.grid(True)

# Menampilkan plot di Streamlit
st.pyplot(fig)


"""**Analisis:**

Grafik elbow biasanya menunjukkan titik di mana penurunan inertia mulai melambat secara signifikan (titik "elbow")

**Dari hasil ini:**

Inertia menurun tajam dari k=2 ke k=3 dan mulai melambat setelah k=3.
Penurunan inertia setelah k=3 menjadi lebih kecil, sehingga k=3 dapat dianggap titik elbow.

**b. Silhouette Score**
"""

# Daftar jumlah cluster yang diuji
k_values = range(2, 11)  # Misalnya 2 sampai 10 cluster

# Menghitung Silhouette Score untuk setiap jumlah cluster
silhouette_scores = []

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    cluster_labels = kmeans.fit_predict(df_scaled)
    silhouette_avg = silhouette_score(df_scaled, cluster_labels)
    silhouette_scores.append(silhouette_avg)

# Visualisasi Silhouette Scores
st.subheader('Silhouette Score untuk Menentukan Jumlah Cluster Optimal')

fig, ax = plt.subplots(figsize=(10, 6))
sns.lineplot(x=k_values, y=silhouette_scores, marker='o', color='blue', ax=ax)
ax.set_title('Silhouette Score for Optimal k')
ax.set_xlabel('Number of Clusters (k)')
ax.set_ylabel('Silhouette Score')
ax.set_xticks(k_values)
ax.grid(True)

# Menampilkan plot di Streamlit
st.pyplot(fig)

# Menentukan k optimal
optimal_k = k_values[np.argmax(silhouette_scores)]
st.write(f'Optimal number of clusters based on silhouette score: {optimal_k}')

"""**Analisis:**

Nilai silhouette score menunjukkan kualitas clustering, di mana nilai mendekati 1 berarti cluster lebih terpisah dan kompak.

**Dari hasil ini:**

Nilai silhouette score tertinggi adalah pada k=2 (0.6005).
Setelah k=2, silhouette score menurun tetapi tetap relatif stabil di sekitar 0.53–0.57.
Meski k=2 memiliki nilai tertinggi, memilih k=2 mungkin terlalu sederhana untuk dataset yang kompleks.

# **Kesimpulan:**

k=3 adalah pilihan yang lebih baik karena memiliki keseimbangan antara penurunan inertia (dari Elbow Method) dan silhouette score yang masih cukup baik (0.5728).

# 🖥️**7. K-Means Clustering & Visualisasi**👩‍💻
"""

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Jumlah cluster yang ditentukan
optimal_k = 3
st.write(f"Jumlah cluster optimal: {optimal_k}")

# K-Means clustering
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
df['Cluster'] = kmeans.fit_predict(df_scaled)

# Menampilkan pusat cluster (centroid)
st.write("\nPusat cluster (centroid):")
st.write(kmeans.cluster_centers_)

# Menampilkan jumlah data di masing-masing cluster
st.write("\nJumlah data di setiap cluster:")
st.write(df['Cluster'].value_counts())

# Statistik deskriptif untuk setiap cluster
st.write("\nStatistik deskriptif untuk setiap cluster:")
st.write(df.groupby('Cluster').mean())

# Visualisasi hasil clustering (2D projection menggunakan PCA jika data >2D)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(df_scaled)

# Plot hasil clustering
fig, ax = plt.subplots(figsize=(8, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=df['Cluster'], palette='rainbow', s=50, ax=ax)
ax.set_title('K-Means Clustering Visualization')
ax.set_xlabel('Economical Factors')
ax.set_ylabel('Behavioral Factors')
ax.legend(title='Cluster')
ax.grid()

# Menampilkan plot di Streamlit
st.pyplot(fig)

"""# 🛒**8.Analisis Clustering Berdasarkan Spending Behavior**🕵️‍♀️

# 🎲**1.Cluster 0: Pelanggan dengan Pengeluaran Tinggi**

Rata-rata Usia: 56.42 tahun

Rata-rata Pendapatan: 61,747.57

Rata-rata Skor Pengeluaran: 12,452.62

Rata-rata Keanggotaan: 10.14 tahun

Rata-rata Frekuensi Pembelian: 52.63

Rata-rata Jumlah Pembelian Terakhir: 6,276.82

# 🛒Karakteristik:

* Pelanggan ini cenderung lebih tua (rata-rata 56 tahun) dengan pendapatan yang lebih tinggi (rata-rata 61,747).
* Mereka memiliki skor pengeluaran yang sangat tinggi, menunjukkan bahwa mereka cenderung berbelanja lebih banyak.
* Mereka juga lebih lama menjadi anggota perusahaan (rata-rata keanggotaan 10 tahun) dan melakukan pembelian lebih sering (52.63 kali).
* Dengan jumlah pembelian terakhir yang tinggi (6,276.82), mereka bisa dikategorikan sebagai Pelanggan dengan Pengeluaran Tinggi.

# 🪄Strategi Marketing:

Perusahaan dapat menawarkan produk premium atau eksklusif kepada kelompok ini, serta memberikan layanan pelanggan khusus dan diskon besar untuk pembelian jumlah besar. Pelanggan ini sangat potensial untuk penjualan besar-besaran dan loyalitas jangka panjang.

# 🎲**2.Cluster 1: Pelanggan dengan Pengeluaran Hemat**

Rata-rata Usia: 27.02 tahun

Rata-rata Pendapatan: 31,820.48

Rata-rata Skor Pengeluaran: 6,465.77

Rata-rata Keanggotaan: 8.18 tahun

Rata-rata Frekuensi Pembelian: 42.90

Rata-rata Jumlah Pembelian Terakhir: 3,281.03

# 🛒Karakteristik:

* Pelanggan ini lebih muda (rata-rata 27 tahun) dengan pendapatan yang lebih rendah dibandingkan dengan Cluster 0 (rata-rata 31,820).
* Mereka memiliki skor pengeluaran yang lebih rendah (rata-rata 6,465), dan frekuensi pembelian mereka juga lebih rendah (42.90 kali).
* Jumlah pembelian terakhir mereka juga relatif lebih rendah (3,281.03), yang menunjukkan kebiasaan belanja yang lebih moderat.
* Mereka lebih cenderung dapat dikategorikan sebagai Pelanggan Biasa, yang tidak berbelanja terlalu banyak, namun masih aktif.

# 🪄Strategi Marketing:

Strategi pemasaran dapat mencakup penawaran produk dengan harga menengah dan paket diskon yang menekankan efisiensi atau nilai. Selain itu, perusahaan dapat menawarkan berbagai pilihan pembayaran atau program loyalitas untuk meningkatkan pembelian berulang.

# 🎲**3.Cluster 1: Pelanggan dengan Pengeluaran Biasa**

Rata-rata Usia: 41.69 tahun

Rata-rata Pendapatan: 46,681.01

Rata-rata Skor Pengeluaran: 9,437.15

Rata-rata Keanggotaan: 9.84 tahun

Rata-rata Frekuensi Pembelian: 51.23

Rata-rata Jumlah Pembelian Terakhir: 4,766.68

# 🛒Karakteristik:

* Pelanggan ini berada di usia rata-rata sekitar 41 tahun dengan pendapatan yang lebih tinggi (rata-rata 46,681).
* Skor pengeluaran mereka juga moderat (9,437), meskipun lebih tinggi dibandingkan dengan Cluster 1.
* Mereka memiliki frekuensi pembelian yang cukup tinggi (51.23 kali) dan jumlah pembelian terakhir yang relatif tinggi (4,766.68).
* Mereka dapat dianggap sebagai Pelanggan Hemat, yang mungkin berbelanja lebih sering tetapi tidak menghabiskan terlalu banyak dalam setiap transaksi.

# 🪄Strategi Marketing:

Kelompok ini memerlukan insentif yang lebih kecil namun menarik, seperti diskon atau promosi untuk mendorong frekuensi pembelian mereka. Perusahaan dapat fokus pada peningkatan pengalaman berbelanja agar pelanggan ini lebih sering membeli.
"""