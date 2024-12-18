import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score

# 1. Import & Load Dataset
st.title('Customer Behavior Segmentation Analysis')
df = pd.read_csv('Customer Purchase Data.csv')
st.write(df.head())

# 2. Eksplorasi Dataset
st.subheader('Dataset Information')
st.write(df.info())

# Cek Missing Values
st.subheader('Missing Values')
st.write(df.isnull().sum())

# Cek Data Duplikat
st.subheader('Duplicate Rows')
st.write(f"Jumlah duplikasi: {df.duplicated().sum()}")
df = df.drop_duplicates()

# Statistik Deskriptif
st.subheader('Descriptive Statistics')
st.write(df.describe())

# Penghapusan kolom 'Number'
df.drop(["Number"], axis=1, inplace=True)
st.write(df.columns)

# 3. Visualisasi Distribusi Variabel Numerik
st.subheader('Visualisasi Distribusi Variabel Numerik')
# Data Cleaning
df_cleaned = df.dropna()
df_cleaned = df_cleaned.drop_duplicates()

numerical_columns = ['Age', 'Income', 'Spending_Score', 'Membership_Years', 'Purchase_Frequency', 'Last_Purchase_Amount']
for var in numerical_columns:
    if var in df.columns:
        st.write(f"Menampilkan distribusi untuk kolom: {var}")
        if pd.api.types.is_numeric_dtype(df[var]):
            df[var].dropna(inplace=True)
            fig, ax = plt.subplots()
            sns.histplot(df[var], kde=True, bins=30, ax=ax)
            ax.set_title(f'Distribusi {var}')
            st.pyplot(fig)

# Boxplot untuk Visualisasi Outlier
fig, ax = plt.subplots(figsize=(15, 20))
df.plot(kind='box', subplots=True, layout=(6, 3), sharex=False, sharey=False, figsize=(15, 20), ax=ax)
plt.tight_layout()
st.pyplot(fig)

# Korelasi Data dengan Heatmap
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df_cleaned[numerical_columns]), columns=numerical_columns)
correlation_matrix = df_scaled.corr()

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.4f', linewidths=0.5, ax=ax)
ax.set_title('Matriks Korelasi antar Variabel')
st.pyplot(fig)

# 4. Dimensionality Reduction
# Deteksi dan penanganan outlier menggunakan IQR
def handle_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[column] = np.where(df[column] > upper_bound, upper_bound,
                         np.where(df[column] < lower_bound, lower_bound, df[column]))
    return df

for column in numerical_columns:
    df = handle_outliers(df, column)

fig, ax = plt.subplots(figsize=(15, 20))
df.plot(kind='box', subplots=True, layout=(6, 3), sharex=False, sharey=False, figsize=(15, 20), ax=ax)
plt.tight_layout()
st.pyplot(fig)

# 5. PCA (Principal Component Analysis)
pca = PCA(n_components=2)
principal_components = pca.fit_transform(df_scaled[numerical_columns])

pca_data = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
st.write("Hasil PCA (Komponen Utama):")
st.write(pca_data)

# 6. K-Means Clustering & Visualisasi
# Menentukan jumlah cluster yang optimal dengan Elbow Method
inertia = []
k_values = range(2, 10)

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(df_scaled)
    inertia.append(kmeans.inertia_)

fig, ax = plt.subplots(figsize=(10, 6))
sns.lineplot(x=k_values, y=inertia, marker='o', ax=ax)
ax.set_title('Elbow Method for Optimal k')
ax.set_xlabel('Number of Clusters (k)')
ax.set_ylabel('Inertia')
ax.grid()
st.pyplot(fig)

# Menghitung Silhouette Score
silhouette_scores = []
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(df_scaled)
    cluster_labels = kmeans.labels_
    silhouette_avg = silhouette_score(df_scaled, cluster_labels)
    silhouette_scores.append(silhouette_avg)

fig, ax = plt.subplots(figsize=(10, 6))
sns.lineplot(x=k_values, y=silhouette_scores, marker='o', ax=ax)
ax.set_title('Silhouette Scores for Different Numbers of Clusters')
ax.set_xlabel('Number of Clusters (k)')
ax.set_ylabel('Silhouette Score')
ax.grid()
st.pyplot(fig)

optimal_k = k_values[np.argmax(silhouette_scores)]
st.write(f'Optimal number of clusters based on silhouette score: {optimal_k}')

# 7. K-Means Clustering dan Visualisasi
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
df_scaled['Cluster'] = kmeans.fit_predict(df_scaled)

# Visualisasi Clustering Hasil PCA
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(x=pca_data['PC1'], y=pca_data['PC2'], hue=df_scaled['Cluster'], palette='Set1', ax=ax)
ax.set_title('Clustering Results (PCA)')
st.pyplot(fig)
