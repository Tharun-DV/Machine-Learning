import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def perform_clustering(data, n_clusters, method):
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(data)

    if method == "K-Means":
        model = KMeans(n_clusters=n_clusters, random_state=42)
        model.fit(features_scaled)
        labels = model.labels_
    elif method == "EM":
        model = GaussianMixture(n_components=n_clusters, random_state=42)
        model.fit(features_scaled)
        labels = model.predict(features_scaled)

    silhouette = silhouette_score(features_scaled, labels)
    return labels, silhouette


def plot_clusters(data, labels):
    pca = PCA(n_components=2)
    features_pca = pca.fit_transform(data)

    plt.figure(figsize=(8, 6))
    plt.scatter(
        features_pca[:, 0], features_pca[:, 1], c=labels, cmap="viridis", marker="o"
    )
    plt.title("Cluster Visualization")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.colorbar(label="Cluster Label")
    st.pyplot(plt)


st.title("Patient Clustering App")

uploaded_file = st.file_uploader("Upload your medical records CSV file", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Data Preview:")
    st.dataframe(data.head())

    features = st.multiselect(
        "Select features for clustering", options=data.columns.tolist()
    )

    if features:
        data_encoded = pd.get_dummies(data[features], drop_first=True)

        n_clusters = st.slider(
            "Select number of clusters", min_value=2, max_value=10, value=3
        )
        method = st.selectbox("Select clustering method", options=["K-Means", "EM"])

        if st.button("Perform Clustering"):
            labels, silhouette = perform_clustering(data_encoded, n_clusters, method)
            data["Cluster"] = labels
            st.write(f"Silhouette Score: {silhouette:.2f}")
            st.write("Clustered Data:")
            st.dataframe(data)

            plot_clusters(data_encoded, labels)
