# -*- coding: utf-8 -*-
"""
Customer Behavior Segmentation Analysis
====================================

Authors:
--------
1. Pujiani Rahayu Agustin - 24060122130067
2. Meyta Rizki Khairunisa - 24060122130085
3. Aura Arfannisa Az Zahra - 24060122130097
4. Nabila Betari Anjani - 24060122140169
"""

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
import re

def load_and_prepare_data(file_path):
    """Load and prepare the dataset."""
    df = pd.read_csv(file_path)
    print(f"Dataset size: {df.size}")
    
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Drop unnecessary columns
    if 'Number' in df.columns:
        df.drop(['Number'], axis=1, inplace=True)
    
    return df

def plot_distributions(df, variables):
    """Plot distributions for numerical variables."""
    for var in variables:
        if var in df.columns and pd.api.types.is_numeric_dtype(df[var]):
            plt.figure(figsize=(10, 6))
            sns.histplot(df[var].dropna(), kde=True, bins=30)
            plt.title(f'Distribution of {var}')
            plt.show()

def plot_boxplots(df):
    """Plot boxplots for numerical variables."""
    df.plot(kind='box',
            subplots=True,
            layout=(6, 3),
            sharex=False,
            sharey=False,
            figsize=(15, 20))
    plt.tight_layout()
    plt.show()

def plot_correlation_heatmap(df_scaled, numerical_columns):
    """Plot correlation heatmap."""
    correlation_matrix = df_scaled[numerical_columns].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.4f', linewidths=0.5)
    plt.title('Correlation Matrix between Variables')
    plt.show()

def handle_outliers(df, numerical_columns):
    """Handle outliers using IQR method."""
    for column in numerical_columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[column] = np.where(df[column] > upper_bound, upper_bound,
                            np.where(df[column] < lower_bound, lower_bound, df[column]))
    return df

def perform_pca_analysis(df_scaled, numerical_columns):
    """Perform PCA analysis."""
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(df_scaled[numerical_columns])
    
    # Create DataFrame with PCA results
    components_df = pd.DataFrame(
        pca.components_,
        columns=df_scaled[numerical_columns].columns,
        index=['PC1', 'PC2']
    )
    
    print("\nPrincipal Components Analysis Results:")
    for i, row in components_df.iterrows():
        print(f"\nComponent contributors for {i}:")
        contributing_features = row.index[row.abs() > 0.2]
        print(", ".join(contributing_features))
    
    return principal_components, pca

def find_optimal_clusters(df_scaled, k_range):
    """Find optimal number of clusters using Elbow method and Silhouette score."""
    inertias = []
    silhouette_scores = []
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        cluster_labels = kmeans.fit_predict(df_scaled)
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(df_scaled, cluster_labels))
    
    # Plot Elbow curve
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, inertias, 'bo-')
    plt.title('Elbow Method for Optimal k')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia')
    plt.grid()
    plt.show()
    
    # Plot Silhouette scores
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, silhouette_scores, 'ro-')
    plt.title('Silhouette Score for Optimal k')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.grid()
    plt.show()
    
    return inertias, silhouette_scores

def perform_clustering(df, df_scaled, optimal_k):
    """Perform K-means clustering and visualize results."""
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    df['Cluster'] = kmeans.fit_predict(df_scaled)
    
    # PCA for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(df_scaled)
    
    # Plot clustering results
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df['Cluster'], 
                         cmap='rainbow', s=50)
    plt.title('K-Means Clustering Visualization')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.colorbar(scatter, label='Cluster')
    plt.grid(True)
    plt.show()
    
    return kmeans

def main():
    # Load and prepare data
    df = load_and_prepare_data('Customer Purchase Data.csv')
    
    # Define numerical columns
    numerical_columns = ['Age', 'Income', 'Spending_Score', 'Membership_Years',
                        'Purchase_Frequency', 'Last_Purchase_Amount']
    
    # Plot distributions
    plot_distributions(df, numerical_columns)
    
    # Plot boxplots before outlier handling
    print("Boxplots before outlier handling:")
    plot_boxplots(df)
    
    # Handle outliers
    df = handle_outliers(df, numerical_columns)
    
    # Plot boxplots after outlier handling
    print("Boxplots after outlier handling:")
    plot_boxplots(df)
    
    # Scale the data
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df[numerical_columns]),
                            columns=numerical_columns)
    
    # Plot correlation heatmap
    plot_correlation_heatmap(df_scaled, numerical_columns)
    
    # Perform PCA analysis
    principal_components, pca = perform_pca_analysis(df_scaled, numerical_columns)
    
    # Find optimal number of clusters
    k_range = range(2, 10)
    inertias, silhouette_scores = find_optimal_clusters(df_scaled, k_range)
    
    # Perform clustering with optimal k=3
    optimal_k = 3
    kmeans = perform_clustering(df, df_scaled, optimal_k)
    
    # Print cluster statistics
    print("\nCluster Statistics:")
    print(df.groupby('Cluster').mean())

if __name__ == "__main__":
    main()