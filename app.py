import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.ensemble import RandomForestClassifier
import joblib
import io
from scipy import stats

st.set_page_config(page_title="Customer Segmentation", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv("Customer Purchase Data.csv")

def remove_outliers(df, columns, method='zscore', threshold=3):
    df_clean = df.copy()
    for column in columns:
        if method == 'zscore':
            z_scores = stats.zscore(df_clean[column])
            df_clean = df_clean[abs(z_scores) < threshold]
        elif method == 'iqr':
            Q1 = df_clean[column].quantile(0.25)
            Q3 = df_clean[column].quantile(0.75)
            IQR = Q3 - Q1
            df_clean = df_clean[
                (df_clean[column] >= Q1 - threshold * IQR) & 
                (df_clean[column] <= Q3 + threshold * IQR)
            ]
    return df_clean

try:
    # Load data
    df = load_data()
    
    st.title("Customer Segmentation Analysis")
    
    # 1. Data Overview
    st.header("1. Data Overview")
    st.write("Shape:", df.shape)
    st.write("Sample data:", df.head())
    
    # Basic stats
    st.write("Summary statistics:", df.describe())
    
    # Missing values
    missing = pd.DataFrame({
        'Missing Values': df.isnull().sum(),
        'Percentage': (df.isnull().sum() / len(df) * 100)
    })
    st.write("Missing values:", missing)
    
    # 2. Feature Selection
    st.header("2. Feature Selection")
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    variables = st.multiselect(
        "Select Features for Clustering:",
        numerical_cols,
        default=['Income', 'Spending_Score']
    )
    
    if len(variables) >= 2:
        # 3. Preprocessing
        st.header("3. Preprocessing")
        
        # Outlier removal
        outlier_method = st.selectbox("Outlier Detection Method:", ['zscore', 'iqr'])
        threshold = st.slider("Outlier Threshold:", 1.0, 5.0, 3.0)
        df_cleaned = remove_outliers(df, variables, outlier_method, threshold)
        st.write(f"Rows after outlier removal: {len(df_cleaned)} (removed {len(df) - len(df_cleaned)} rows)")
        
        # Scaling
        scaler = StandardScaler()
        df_scaled = scaler.fit_transform(df_cleaned[variables])
        df_scaled = pd.DataFrame(df_scaled, columns=variables)
        
        # Optional PCA
        use_pca = st.checkbox("Apply PCA")
        if use_pca and len(variables) > 2:
            n_components = st.slider("Number of components:", 2, len(variables), 2)
            pca = PCA(n_components=n_components)
            df_pca = pca.fit_transform(df_scaled)
            
            # Plot explained variance
            fig, ax = plt.subplots()
            variance_ratio = pca.explained_variance_ratio_
            ax.bar(range(1, len(variance_ratio) + 1), variance_ratio)
            ax.set_xlabel('Principal Component')
            ax.set_ylabel('Explained Variance Ratio')
            st.pyplot(fig)
            
            clustering_data = df_pca
        else:
            clustering_data = df_scaled
        
        # 4. Clustering
        st.header("4. Clustering")
        algorithm = st.selectbox("Clustering Algorithm:", ["KMeans", "DBSCAN"])
        
        if algorithm == "KMeans":
            # Find optimal k
            max_k = st.slider("Maximum number of clusters:", 2, 10, 6)
            metrics = []
            
            for k in range(2, max_k + 1):
                kmeans = KMeans(n_clusters=k, random_state=42)
                labels = kmeans.fit_predict(clustering_data)
                silhouette = silhouette_score(clustering_data, labels)
                metrics.append({
                    'k': k,
                    'silhouette': silhouette,
                    'inertia': kmeans.inertia_
                })
            
            # Plot metrics
            metrics_df = pd.DataFrame(metrics)
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            
            ax1.plot(metrics_df['k'], metrics_df['inertia'], 'bo-')
            ax1.set_xlabel('Number of Clusters (k)')
            ax1.set_ylabel('Inertia')
            ax1.set_title('Elbow Method')
            
            ax2.plot(metrics_df['k'], metrics_df['silhouette'], 'ro-')
            ax2.set_xlabel('Number of Clusters (k)')
            ax2.set_ylabel('Silhouette Score')
            ax2.set_title('Silhouette Analysis')
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Final clustering
            k = st.slider("Select number of clusters:", 2, max_k, 3)
            model = KMeans(n_clusters=k, random_state=42)
            
        else:  # DBSCAN
            eps = st.slider("eps:", 0.1, 2.0, 0.5)
            min_samples = st.slider("min_samples:", 2, 10, 5)
            model = DBSCAN(eps=eps, min_samples=min_samples)
        
        # Fit model and get labels
        labels = model.fit_predict(clustering_data)
        
        # 5. Results
        st.header("5. Results")
        
        # Add cluster labels to original data
        df_cleaned['Cluster'] = labels
        
        # Basic cluster stats
        cluster_sizes = pd.DataFrame({
            'Size': df_cleaned['Cluster'].value_counts(),
            'Percentage': df_cleaned['Cluster'].value_counts(normalize=True) * 100
        })
        st.write("Cluster sizes:", cluster_sizes)
        
        # Cluster means
        cluster_means = df_cleaned.groupby('Cluster')[variables].mean()
        st.write("Cluster means:", cluster_means)
        
        # Visualization
        if len(variables) >= 2:
            fig, ax = plt.subplots(figsize=(10, 6))
            scatter = ax.scatter(
                df_cleaned[variables[0]],
                df_cleaned[variables[1]],
                c=labels,
                cmap='viridis'
            )
            plt.colorbar(scatter)
            ax.set_xlabel(variables[0])
            ax.set_ylabel(variables[1])
            st.pyplot(fig)
        
        # Save model
        if st.button("Save Model"):
            model_data = {
                'model': model,
                'scaler': scaler,
                'features': variables
            }
            joblib.dump(model_data, 'clustering_model.pkl')
            st.success("Model saved successfully!")

    else:
        st.warning("Please select at least 2 features for clustering.")

except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    st.write("Please check your data and try again.")