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

# Set style for better visualization
plt.style.use('seaborn')

# Utility Functions
def remove_outliers(df, columns, method='zscore', threshold=3):
    """
    Remove outliers using either Z-score or IQR method
    """
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

# Load Data
try:
    df = pd.read_csv("Customer Purchase Data.csv")
    
    # ============= EXPLORATORY DATA ANALYSIS =============
    st.title("Advanced Customer Segmentation Analysis")
    st.header("1. Exploratory Data Analysis (EDA)")
    
    # 1.1 Dataset Overview
    st.subheader("1.1 Dataset Overview")
    st.write("Shape of dataset:", df.shape)
    st.write("First 5 rows of dataset:")
    st.write(df.head())
    
    # 1.2 Data Info and Missing Values
    st.subheader("1.2 Data Information and Missing Values")
    col1, col2 = st.columns(2)
    
    with col1:
        buffer = io.StringIO()
        df.info(buf=buffer)
        st.text(buffer.getvalue())
    
    with col2:
        missing_df = pd.DataFrame({
            'Missing Values': df.isnull().sum(),
            'Percentage': (df.isnull().sum() / len(df) * 100)
        })
        st.write("Missing Values Analysis:")
        st.write(missing_df)
    
    # 1.3 Statistical Summary
    st.subheader("1.3 Statistical Summary")
    st.write(df.describe())
    
    # 1.4 Distribution Analysis
    st.subheader("1.4 Distribution Analysis")
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    
    for col in numerical_cols:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Histogram with KDE
        sns.histplot(data=df, x=col, kde=True, ax=ax1)
        ax1.set_title(f'Distribution of {col}')
        
        # Box Plot
        sns.boxplot(y=df[col], ax=ax2)
        ax2.set_title(f'Box Plot of {col}')
        
        st.pyplot(fig)
        plt.close()
        
        # Distribution statistics
        stats_df = pd.DataFrame({
            'Metric': ['Mean', 'Median', 'Std Dev', 'Skewness', 'Kurtosis'],
            'Value': [
                df[col].mean(),
                df[col].median(),
                df[col].std(),
                stats.skew(df[col]),
                stats.kurtosis(df[col])
            ]
        })
        st.write(f"Statistics for {col}:")
        st.write(stats_df)
    
    # 1.5 Correlation Analysis
    st.subheader("1.5 Correlation Analysis")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df[numerical_cols].corr(), annot=True, cmap='coolwarm', ax=ax)
    plt.title('Correlation Heatmap')
    st.pyplot(fig)
    plt.close()
    
    # ============= FEATURE SELECTION & PREPROCESSING =============
    st.header("2. Feature Selection & Preprocessing")
    
    # 2.1 Feature Selection
    variables = st.multiselect(
        "Select Variables for Clustering:",
        numerical_cols,
        default=['Income', 'Spending_Score', 'Membership_Years']
    )
    
    if len(variables) >= 2:
        # 2.2 Outlier Treatment
        st.subheader("2.2 Outlier Treatment")
        outlier_method = st.selectbox(
            "Select Outlier Detection Method:",
            ['zscore', 'iqr']
        )
        outlier_threshold = st.slider(
            "Select Outlier Threshold:",
            min_value=1.0,
            max_value=5.0,
            value=3.0,
            step=0.5
        )
        
        df_cleaned = remove_outliers(df, variables, method=outlier_method, threshold=outlier_threshold)
        st.write(f"Rows removed: {len(df) - len(df_cleaned)}")
        
        # 2.3 Scaling
        st.subheader("2.3 Data Scaling")
        scaling_method = st.selectbox(
            "Select Scaling Method:",
            ['Standard', 'Robust']
        )
        
        if scaling_method == 'Standard':
            scaler = StandardScaler()
        else:
            scaler = RobustScaler()
            
        df_scaled = scaler.fit_transform(df_cleaned[variables])
        df_scaled = pd.DataFrame(df_scaled, columns=variables)
        
        # Show scaling results
        col1, col2 = st.columns(2)
        with col1:
            st.write("Before Scaling:")
            st.write(df_cleaned[variables].describe())
        
        with col2:
            st.write("After Scaling:")
            st.write(df_scaled.describe())
        
        # 2.4 PCA
        st.subheader("2.4 Dimensionality Reduction")
        use_pca = st.checkbox("Apply PCA", value=False)
        
        if use_pca and len(variables) > 2:
            n_components = st.slider(
                "Select number of components:",
                min_value=2,
                max_value=len(variables),
                value=min(3, len(variables))
            )
            
            pca = PCA(n_components=n_components)
            df_pca = pca.fit_transform(df_scaled)
            
            # Show explained variance
            fig, ax = plt.subplots(figsize=(10, 6))
            explained_variance = pca.explained_variance_ratio_
            ax.bar(range(1, len(explained_variance) + 1), explained_variance)
            ax.plot(range(1, len(explained_variance) + 1), 
                   np.cumsum(explained_variance), 
                   'r-o')
            ax.set_xlabel('Principal Components')
            ax.set_ylabel('Explained Variance Ratio')
            plt.title('PCA Explained Variance')
            st.pyplot(fig)
            plt.close()
            
            clustering_data = df_pca
        else:
            clustering_data = df_scaled
        
        # ============= CLUSTERING ANALYSIS =============
        st.header("3. Clustering Analysis")
        
        # Algorithm Selection
        clustering_algo = st.selectbox(
            "Select Clustering Algorithm",
            ["KMeans", "GaussianMixture", "DBSCAN"]
        )
        
        if clustering_algo in ["KMeans", "GaussianMixture"]:
            # Elbow Method and Silhouette Analysis
            max_clusters = st.slider("Maximum number of clusters to try:", 2, 15, 10)
            
            metrics = {
                'n_clusters': list(range(2, max_clusters + 1)),
                'silhouette': [],
                'davies_bouldin': [],
                'calinski_harabasz': []
            }
            
            if clustering_algo == "KMeans":
                metrics['inertia'] = []
            
            for k in range(2, max_clusters + 1):
                if clustering_algo == "KMeans":
                    model = KMeans(n_clusters=k, n_init=10, random_state=42)
                else:
                    model = GaussianMixture(n_components=k, random_state=42)
                
                labels = model.fit_predict(clustering_data)
                
                if clustering_algo == "KMeans":
                    metrics['inertia'].append(model.inertia_)
                    
                metrics['silhouette'].append(silhouette_score(clustering_data, labels))
                metrics['davies_bouldin'].append(davies_bouldin_score(clustering_data, labels))
                metrics['calinski_harabasz'].append(calinski_harabasz_score(clustering_data, labels))
            
            # Plot metrics
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Clustering Metrics')
            
            if clustering_algo == "KMeans":
                axes[0, 0].plot(metrics['n_clusters'], metrics['inertia'], 'bo-')
                axes[0, 0].set_title('Elbow Method')
                axes[0, 0].set_xlabel('Number of Clusters')
                axes[0, 0].set_ylabel('Inertia')
            
            axes[0, 1].plot(metrics['n_clusters'], metrics['silhouette'], 'ro-')
            axes[0, 1].set_title('Silhouette Score')
            axes[0, 1].set_xlabel('Number of Clusters')
            axes[0, 1].set_ylabel('Score')
            
            axes[1, 0].plot(metrics['n_clusters'], metrics['davies_bouldin'], 'go-')
            axes[1, 0].set_title('Davies-Bouldin Score')
            axes[1, 0].set_xlabel('Number of Clusters')
            axes[1, 0].set_ylabel('Score')
            
            axes[1, 1].plot(metrics['n_clusters'], metrics['calinski_harabasz'], 'mo-')
            axes[1, 1].set_title('Calinski-Harabasz Score')
            axes[1, 1].set_xlabel('Number of Clusters')
            axes[1, 1].set_ylabel('Score')
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
            # Select optimal number of clusters
            optimal_k = st.slider(
                "Select Number of Clusters:",
                min_value=2,
                max_value=max_clusters,
                value=3
            )
            
            if clustering_algo == "KMeans":
                final_model = KMeans(
                    n_clusters=optimal_k,
                    n_init=10,
                    random_state=42
                )
            else:
                final_model = GaussianMixture(
                    n_components=optimal_k,
                    random_state=42
                )
        
        else:  # DBSCAN
            eps = st.slider("Select eps value:", 0.1, 2.0, 0.5)
            min_samples = st.slider("Select min_samples:", 2, 10, 5)
            final_model = DBSCAN(eps=eps, min_samples=min_samples)
        
        # Fit final model and get labels
        labels = final_model.fit_predict(clustering_data)
        df_cleaned['Cluster'] = labels
        
        # ============= CLUSTER ANALYSIS & VISUALIZATION =============
        st.header("4. Cluster Analysis & Visualization")
        
        # 4.1 Cluster Statistics
        st.subheader("4.1 Cluster Statistics")
        for cluster in sorted(set(labels)):
            st.write(f"### Cluster {cluster}")
            cluster_data = df_cleaned[df_cleaned['Cluster'] == cluster]
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("Size:", len(cluster_data))
                st.write("Percentage:", f"{(len(cluster_data) / len(df_cleaned) * 100):.2f}%")
            
            with col2:
                st.write("Statistics:")
                st.write(cluster_data[variables].describe())
        
        # 4.2 Feature Importance
        st.subheader("4.2 Feature Importance Analysis")
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(df_scaled, labels)
        
        feature_importance = pd.DataFrame({
            'feature': variables,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=feature_importance, x='importance', y='feature', ax=ax)
        plt.title('Feature Importance for Clustering')
        st.pyplot(fig)
        plt.close()
        
        # 4.3 Cluster Visualization
        st.subheader("4.3 Cluster Visualization")
        
        if len(variables) >= 2:
            x_var = st.selectbox("Select X variable:", variables)
            y_var = st.selectbox("Select Y variable:", variables, index=1)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            scatter = ax.scatter(df_cleaned[x_var], 
                               df_cleaned[y_var], 
                               c=labels, 
                               cmap='viridis')
            plt.colorbar(scatter)
            ax.set_xlabel(x_var)
            ax.set_ylabel(y_var)
            plt.title('Cluster Visualization')
            st.pyplot(fig)
            plt.close()
        
        # Save model if requested
        if st.button("Save Model"):
            model_data = {
                'model': final_model,
                'scaler': scaler,
                'variables': variables,
                'pca': pca if use_pca else None
            }
            joblib.dump(model_data, 'customer_segmentation_model.pkl')
            st.success("Model saved successfully!")

except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    st.write("Please ensure your dataset contains the required columns and is properly formatted.")