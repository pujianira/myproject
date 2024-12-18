import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import joblib
import io

# Konfigurasi Streamlit
st.set_page_config(page_title="Customer Segmentation", layout="wide")
st.title("Customer Behavior Segmentation Analysis")

# Load dataset
try:
    df = pd.read_csv("Customer Purchase Data.csv")
    
    # ============= EXPLORATORY DATA ANALYSIS =============
    st.header("1. Exploratory Data Analysis (EDA)")
    
    # 1.1 Dataset Overview
    st.subheader("1.1 Dataset Overview")
    st.write("Shape of dataset:", df.shape)
    st.write("### First 5 rows of dataset")
    st.write(df.head())
    
    # 1.2 Data Info
    st.subheader("1.2 Data Information")
    buffer = io.StringIO()
    df.info(buf=buffer)
    st.text(buffer.getvalue())
    
    # 1.3 Statistical Summary
    st.subheader("1.3 Statistical Summary")
    st.write(df.describe())
    
    # 1.4 Missing Values Analysis
    st.subheader("1.4 Missing Values Analysis")
    missing_values = df.isnull().sum()
    missing_percent = (missing_values / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing Values': missing_values,
        'Percentage': missing_percent
    })
    st.write(missing_df)
    
    # 1.5 Distribution of All Numerical Variables
    st.subheader("1.5 Distribution of Numerical Variables")
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    
    for col in numerical_cols:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Histogram
        sns.histplot(data=df, x=col, kde=True, ax=ax1)
        ax1.set_title(f'Distribution of {col}')
        
        # Box Plot
        sns.boxplot(data=df, y=col, ax=ax2)
        ax2.set_title(f'Box Plot of {col}')
        
        st.pyplot(fig)
        plt.close()
        
        # Basic statistics
        st.write(f"**Statistics for {col}:**")
        stats_df = pd.DataFrame({
            'Metric': ['Mean', 'Median', 'Std Dev', 'Skewness', 'Kurtosis'],
            'Value': [
                df[col].mean(),
                df[col].median(),
                df[col].std(),
                df[col].skew(),
                df[col].kurtosis()
            ]
        })
        st.write(stats_df)
    
    # 1.6 Correlation Analysis
    st.subheader("1.6 Correlation Analysis")
    
    # Correlation Matrix
    corr_matrix = df[numerical_cols].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
    plt.title('Correlation Matrix')
    st.pyplot(fig)
    plt.close()
    
    # ============= FEATURE SELECTION & PREPROCESSING =============
    st.header("2. Feature Selection & Preprocessing")
    
    # 2.1 Select Features
    st.subheader("2.1 Feature Selection")
    variables = st.multiselect(
        "Select Variables for Clustering:",
        df.columns,
        default=['Income', 'Spending_Score', 'Membership_Years'],
        help="Choose features that best represent customer behavior"
    )
    
    if len(variables) >= 2:
        # 2.2 Normalization
        st.subheader("2.2 Data Normalization")
        scaler = StandardScaler()
        df_scaled = scaler.fit_transform(df[variables])
        df_scaled = pd.DataFrame(df_scaled, columns=variables)
        
        st.write("### Before Normalization:")
        st.write(df[variables].describe())
        
        st.write("### After Normalization:")
        st.write(df_scaled.describe())
        
        # Visualization of normalized data
        fig, axes = plt.subplots(1, len(variables), figsize=(15, 5))
        for i, col in enumerate(variables):
            sns.histplot(data=df_scaled, x=col, kde=True, ax=axes[i])
            axes[i].set_title(f'Normalized {col}')
        st.pyplot(fig)
        plt.close()
        
        # ============= CLUSTERING ANALYSIS =============
        st.header("3. Clustering Analysis")
        
        # 3.1 Elbow Method
        st.subheader("3.1 Elbow Method")
        inertias = []
        silhouette_scores = []
        k_range = range(2, 11)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Elbow curve
            for k in k_range:
                kmeans = KMeans(n_clusters=k, random_state=42)
                kmeans.fit(df_scaled)
                inertias.append(kmeans.inertia_)
                
                if k > 1:
                    score = silhouette_score(df_scaled, kmeans.labels_)
                    silhouette_scores.append(score)
            
            fig, ax = plt.subplots()
            ax.plot(k_range, inertias, 'bo-')
            ax.set_xlabel('Number of Clusters (k)')
            ax.set_ylabel('Inertia')
            ax.set_title('Elbow Method')
            st.pyplot(fig)
            plt.close()
        
        with col2:
            # Silhouette score
            fig, ax = plt.subplots()
            ax.plot(range(2, 11), silhouette_scores, 'ro-')
            ax.set_xlabel('Number of Clusters (k)')
            ax.set_ylabel('Silhouette Score')
            ax.set_title('Silhouette Score Analysis')
            st.pyplot(fig)
            plt.close()
        
        # 3.2 Optimal Cluster Selection
        optimal_k = st.slider(
            "Select Number of Clusters:",
            min_value=2,
            max_value=10,
            value=3,
            help="Choose based on Elbow Method and Silhouette Score"
        )
        
        # 3.3 Final Clustering
        kmeans = KMeans(n_clusters=optimal_k, random_state=42)
        df['Cluster'] = kmeans.fit_predict(df_scaled)
        
        # ============= CLUSTER ANALYSIS & VISUALIZATION =============
        st.header("4. Cluster Analysis & Visualization")
        
        # 4.1 Cluster Statistics
        st.subheader("4.1 Cluster Statistics")
        cluster_stats = df.groupby('Cluster')[variables].agg(['mean', 'std', 'min', 'max']).round(2)
        st.write(cluster_stats)
        
        # 4.2 Cluster Size Distribution
        st.subheader("4.2 Cluster Size Distribution")
        cluster_sizes = df['Cluster'].value_counts()
        fig, ax = plt.subplots()
        cluster_sizes.plot(kind='bar', ax=ax)
        plt.title('Cluster Size Distribution')
        st.pyplot(fig)
        plt.close()
        
        # 4.3 Cluster Visualization
        st.subheader("4.3 Cluster Visualization")
        
        # Pairplot for all variables
        st.write("#### Pairplot of Variables by Cluster")
        pairplot_fig = sns.pairplot(df, hue='Cluster', vars=variables, palette='Set2')
        st.pyplot(pairplot_fig)
        plt.close()
        
        # 4.4 Cluster Interpretation
        st.subheader("4.4 Cluster Interpretation")
        cluster_behaviors = {}
        
        for cluster in range(optimal_k):
            cluster_data = df[df['Cluster'] == cluster]
            behavior = ""
            
            # Analyze spending behavior
            if 'Spending_Score' in variables:
                avg_spending = cluster_data['Spending_Score'].mean()
                if avg_spending < df['Spending_Score'].quantile(0.33):
                    behavior += "Conservative Spender"
                elif avg_spending > df['Spending_Score'].quantile(0.66):
                    behavior += "High Spender"
                else:
                    behavior += "Moderate Sp"
finally:
    print("Always executed regardless of error.")