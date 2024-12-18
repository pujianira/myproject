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
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

# Konfigurasi Streamlit
st.set_page_config(page_title="Advanced Customer Segmentation", layout="wide")
st.title("Advanced Customer Behavior Segmentation Analysis")

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

def plot_distribution(df, column):
    """
    Create distribution plot using plotly
    """
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=df[column], nbinsx=30, name='Histogram'))
    fig.add_trace(go.Violin(x=df[column], name='Violin Plot', side='positive'))
    
    fig.update_layout(
        title=f'Distribution of {column}',
        xaxis_title=column,
        yaxis_title='Count',
        showlegend=True
    )
    
    return fig

def create_correlation_heatmap(df, cols):
    """
    Create correlation heatmap using plotly
    """
    corr_matrix = df[cols].corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmin=-1, zmax=1,
        text=np.round(corr_matrix.values, 2),
        texttemplate='%{text}',
        textfont={"size": 10},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title='Correlation Heatmap',
        width=800,
        height=800
    )
    
    return fig

# Load Data
try:
    df = pd.read_csv("Customer Purchase Data.csv")
    
    # ============= EXPLORATORY DATA ANALYSIS =============
    st.header("1. Exploratory Data Analysis (EDA)")
    
    # 1.1 Dataset Overview
    st.subheader("1.1 Dataset Overview")
    st.write("Shape of dataset:", df.shape)
    st.write("### First 5 rows of dataset")
    st.write(df.head())
    
    # 1.2 Data Info and Missing Values
    st.subheader("1.2 Data Information and Missing Values")
    col1, col2 = st.columns(2)
    
    with col1:
        buffer = io.StringIO()
        df.info(buf=buffer)
        st.text(buffer.getvalue())
    
    with col2:
        missing_values = df.isnull().sum()
        missing_percent = (missing_values / len(df)) * 100
        missing_df = pd.DataFrame({
            'Missing Values': missing_values,
            'Percentage': missing_percent
        })
        st.write("Missing Values Analysis:")
        st.write(missing_df)
    
    # 1.3 Statistical Summary
    st.subheader("1.3 Statistical Summary")
    st.write(df.describe())
    
    # 1.4 Advanced Distribution Analysis
    st.subheader("1.4 Advanced Distribution Analysis")
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    
    for col in numerical_cols:
        # Distribution statistics
        skewness = stats.skew(df[col])
        kurtosis = stats.kurtosis(df[col])
        
        st.write(f"### Distribution Analysis for {col}")
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(plot_distribution(df, col), use_container_width=True)
        
        with col2:
            stats_df = pd.DataFrame({
                'Metric': ['Mean', 'Median', 'Std Dev', 'Skewness', 'Kurtosis'],
                'Value': [
                    df[col].mean(),
                    df[col].median(),
                    df[col].std(),
                    skewness,
                    kurtosis
                ]
            })
            st.write("Distribution Statistics:")
            st.write(stats_df)
            
            # Normality test
            stat, p_value = stats.normaltest(df[col])
            st.write(f"Normality Test (D'Agostino's K^2):")
            st.write(f"p-value: {p_value:.4f}")
            if p_value < 0.05:
                st.write("Distribution is not normal (p < 0.05)")
            else:
                st.write("Distribution appears normal (p >= 0.05)")
    
    # 1.5 Correlation Analysis
    st.subheader("1.5 Advanced Correlation Analysis")
    st.plotly_chart(create_correlation_heatmap(df, numerical_cols))
    
    # ============= FEATURE SELECTION & PREPROCESSING =============
    st.header("2. Feature Selection & Advanced Preprocessing")
    
    # 2.1 Feature Selection
    st.subheader("2.1 Feature Selection")
    variables = st.multiselect(
        "Select Variables for Clustering:",
        numerical_cols,
        default=['Income', 'Spending_Score', 'Membership_Years'],
        help="Choose features that best represent customer behavior"
    )
    
    if len(variables) >= 2:
        # 2.2 Outlier Handling
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
        
        # 2.4 Dimensionality Reduction
        st.subheader("2.4 Dimensionality Reduction")
        use_pca = st.checkbox("Apply PCA", value=False)
        
        if use_pca:
            n_components = st.slider(
                "Select number of components:",
                min_value=2,
                max_value=len(variables),
                value=min(3, len(variables))
            )
            
            pca = PCA(n_components=n_components)
            df_pca = pca.fit_transform(df_scaled)
            
            # Show explained variance
            explained_variance = pca.explained_variance_ratio_
            cumulative_variance = np.cumsum(explained_variance)
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=list(range(1, len(explained_variance) + 1)),
                y=explained_variance,
                name='Individual'
            ))
            fig.add_trace(go.Scatter(
                x=list(range(1, len(cumulative_variance) + 1)),
                y=cumulative_variance,
                name='Cumulative'
            ))
            
            fig.update_layout(
                title='PCA Explained Variance',
                xaxis_title='Principal Component',
                yaxis_title='Explained Variance Ratio'
            )
            
            st.plotly_chart(fig)
            
            # Use PCA results for clustering
            clustering_data = df_pca
        else:
            clustering_data = df_scaled
        
        # ============= CLUSTERING ANALYSIS =============
        st.header("3. Advanced Clustering Analysis")
        
        # 3.1 Algorithm Selection
        st.subheader("3.1 Clustering Algorithm Selection")
        clustering_algo = st.selectbox(
            "Select Clustering Algorithm",
            ["KMeans", "GaussianMixture", "DBSCAN"]
        )
        
        if clustering_algo in ["KMeans", "GaussianMixture"]:
            # Elbow Method and Silhouette Analysis
            max_clusters = st.slider("Maximum number of clusters to try:", 2, 15, 10)
            
            inertias = []
            silhouette_scores = []
            davies_bouldin_scores = []
            calinski_harabasz_scores = []
            
            for k in range(2, max_clusters + 1):
                if clustering_algo == "KMeans":
                    model = KMeans(n_clusters=k, n_init=10, random_state=42)
                else:
                    model = GaussianMixture(n_components=k, random_state=42)
                
                labels = model.fit_predict(clustering_data)
                
                if hasattr(model, 'inertia_'):
                    inertias.append(model.inertia_)
                
                silhouette_scores.append(silhouette_score(clustering_data, labels))
                davies_bouldin_scores.append(davies_bouldin_score(clustering_data, labels))
                calinski_harabasz_scores.append(calinski_harabasz_score(clustering_data, labels))
            
            # Plot metrics
            fig = go.Figure()
            
            if inertias:
                fig.add_trace(go.Scatter(
                    x=list(range(2, max_clusters + 1)),
                    y=inertias,
                    name='Inertia'
                ))
            
            fig.add_trace(go.Scatter(
                x=list(range(2, max_clusters + 1)),
                y=silhouette_scores,
                name='Silhouette Score'
            ))
            
            fig.add_trace(go.Scatter(
                x=list(range(2, max_clusters + 1)),
                y=davies_bouldin_scores,
                name='Davies-Bouldin Score'
            ))
            
            fig.add_trace(go.Scatter(
                x=list(range(2, max_clusters + 1)),
                y=calinski_harabasz_scores,
                name='Calinski-Harabasz Score'
            ))
            
            fig.update_layout(
                title='Clustering Metrics by Number of Clusters',
                xaxis_title='Number of Clusters',
                yaxis_title='Score'
            )
            
            st.plotly_chart(fig)
            
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
        
        # Fit final model
        labels = final_model.fit_predict(clustering_data)
        
        # Add cluster labels to original data
        df_cleaned['Cluster'] = labels
        
        # ============= CLUSTER ANALYSIS & VISUALIZATION =============
        st.header("4. Cluster Analysis & Visualization")
        
        # 4.1 Cluster Statistics
        st.subheader("4.1 Detailed Cluster Statistics")
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
        
        fig = px.bar(
            feature_importance,
            x='importance',
            y='feature',
            title='Feature Importance for Clustering'
        )
        st.plotly_chart(fig)
        
        # 4.3 Interactive Cluster Visualization
        st.subheader("4.3 Interactive Cluster Visualization")
        
        if len(variables) >= 2:
            x_var = st.selectbox("Select X variable:", variables)
            y_var = st.selectbox("Select Y variable:", variables, index=1)
finally:
    print("Always executed regardless of error.")