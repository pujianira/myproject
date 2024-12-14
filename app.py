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
st.title("Customer Behavior Segmentation Analysis")

# Load dataset
try:
    df = pd.read_csv("Customer Purchase Data.csv")
    st.write("### Dataset Preview")
    st.write(df.head())
    
    # Data quality checks
    st.write("### Data Quality Analysis")
    missing_values = df.isnull().sum()
    st.write("Missing Values:", missing_values)
    
    # Menambahkan deskripsi statistik dasar
    st.write("### Statistical Summary")
    st.write(df.describe())

    # Pilih variabel untuk clustering dengan deskripsi
    st.write("### Feature Selection")
    st.write("Select relevant features for customer behavior analysis:")
    behavior_vars = {
        'Age': 'Customer age in years',
        'Income': 'Annual income in currency',
        'Spending_Score': 'Customer spending behavior score',
        'Last_Purchase_Amount': 'Amount of last purchase',
        'Membership_Years': 'Years as a member',
        'Purchase_Frequency': 'How often customer makes purchases'
    }
    
    variables = st.multiselect(
        "Select Variables for Clustering:",
        df.columns,
        default=['Income', 'Spending_Score', 'Membership_Years'],
        help="Choose features that best represent customer behavior"
    )

    if len(variables) >= 2:
        # Enhanced visualization section
        st.write("### Feature Distribution Analysis")
        cols = st.columns(2)
        for idx, var in enumerate(variables):
            with cols[idx % 2]:
                fig, ax = plt.subplots()
                sns.histplot(df[var], kde=True, bins=30, ax=ax)
                plt.title(f'Distribution of {var}')
                st.pyplot(fig)
                
                # Add basic statistics
                st.write(f"**{var} Statistics:**")
                st.write(f"Mean: {df[var].mean():.2f}")
                st.write(f"Median: {df[var].median():.2f}")
                st.write(f"Std Dev: {df[var].std():.2f}")

        # Correlation analysis
        st.write("### Feature Correlation Analysis")
        corr_matrix = df[variables].corr()
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)

        # Data preprocessing
        scaler = StandardScaler()
        df_scaled = scaler.fit_transform(df[variables])
        df_scaled = pd.DataFrame(df_scaled, columns=variables)

        # Optimal cluster selection
        st.write("### Cluster Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("#### Elbow Method")
            inertias = []
            silhouette_scores = []
            k_range = range(2, 11)
            
            for k in k_range:
                kmeans = KMeans(n_clusters=k, random_state=42)
                kmeans.fit(df_scaled)
                inertias.append(kmeans.inertia_)
                score = silhouette_score(df_scaled, kmeans.labels_)
                silhouette_scores.append(score)
            
            fig, ax = plt.subplots()
            ax.plot(k_range, inertias, 'bo-')
            ax.set_xlabel('Number of Clusters (k)')
            ax.set_ylabel('Inertia')
            st.pyplot(fig)
        
        with col2:
            st.write("#### Silhouette Score Analysis")
            fig, ax = plt.subplots()
            ax.plot(k_range, silhouette_scores, 'ro-')
            ax.set_xlabel('Number of Clusters (k)')
            ax.set_ylabel('Silhouette Score')
            st.pyplot(fig)

        # Interactive cluster selection
        optimal_k = st.slider(
            "Select Number of Clusters:",
            min_value=2,
            max_value=10,
            value=3,
            help="Choose based on Elbow Method and Silhouette Score"
        )

        # Perform clustering
        kmeans = KMeans(n_clusters=optimal_k, random_state=42)
        df['Cluster'] = kmeans.fit_predict(df_scaled)

        # Enhanced cluster analysis
        st.write("### Cluster Insights")
        
        # Cluster statistics
        cluster_stats = df.groupby('Cluster')[variables].agg(['mean', 'std']).round(2)
        st.write("#### Cluster Statistics")
        st.write(cluster_stats)
        
        # Automated cluster interpretation
        st.write("#### Cluster Behavior Classification")
        cluster_behaviors = {}
        
        for cluster in range(optimal_k):
            cluster_data = df[df['Cluster'] == cluster]
            avg_spending = cluster_data['Spending_Score'].mean()
            avg_income = cluster_data['Income'].mean()
            
            if avg_spending < df['Spending_Score'].quantile(0.33):
                behavior = "Conservative Spender"
            elif avg_spending > df['Spending_Score'].quantile(0.66):
                behavior = "High Spender"
            else:
                behavior = "Moderate Spender"
                
            if 'Income' in variables:
                if avg_income > df['Income'].quantile(0.66):
                    behavior += " (High Income)"
                elif avg_income < df['Income'].quantile(0.33):
                    behavior += " (Low Income)"
                else:
                    behavior += " (Middle Income)"
                    
            cluster_behaviors[cluster] = behavior
            
        for cluster, behavior in cluster_behaviors.items():
            st.write(f"Cluster {cluster}: {behavior}")
            
        # Visualization of clusters
        st.write("### Cluster Visualization")
        if len(variables) >= 2:
            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_subplot(111)
            scatter = ax.scatter(df_scaled[variables[0]], 
                               df_scaled[variables[1]],
                               c=df['Cluster'],
                               cmap='viridis')
            plt.xlabel(variables[0])
            plt.ylabel(variables[1])
            plt.colorbar(scatter)
            st.pyplot(fig)

        # Model persistence
        if st.button("Save Clustering Model"):
            model_data = {
                'model': kmeans,
                'scaler': scaler,
                'features': variables,
                'behaviors': cluster_behaviors
            }
            joblib.dump(model_data, 'customer_segmentation_model.pkl')
            st.success("Model and preprocessing pipeline saved successfully!")

        # Prediction interface
        st.write("### Predict New Customer Behavior")
        with st.form("prediction_form"):
            st.write("Enter New Customer Data:")
            input_data = {}
            
            for var in variables:
                input_data[var] = st.number_input(
                    f"Enter {var}:",
                    value=float(df[var].mean()),
                    help=f"Average value: {df[var].mean():.2f}"
                )
            
            submit_button = st.form_submit_button("Predict Behavior")
            
            if submit_button:
                # Prepare input data
                input_df = pd.DataFrame([input_data])
                input_scaled = scaler.transform(input_df)
                
                # Predict cluster
                cluster = kmeans.predict(input_scaled)[0]
                behavior = cluster_behaviors[cluster]
                
                st.success(f"Customer Segment: **{behavior}** (Cluster {cluster})")
                
                # Show comparison with cluster averages
                st.write("#### Comparison with Cluster Averages")
                comparison_df = pd.DataFrame({
                    'Input Values': input_data,
                    'Cluster Average': df[df['Cluster'] == cluster][variables].mean()
                }).round(2)
                st.write(comparison_df)

    else:
        st.warning("Please select at least 2 variables for clustering analysis.")

except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    st.write("Please ensure your dataset contains the required columns and is properly formatted.")