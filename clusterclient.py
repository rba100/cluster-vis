import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from vectordbclient import get_closest_words;
import streamlit as st

def get_clusters(_conn, algorithm, vectors, n_clusters, random_state=None, distance_threshold=None):
        switcher = {
            "KMeans": lambda: get_clusters_kmeans(_conn, vectors, n_clusters, random_state=random_state),
            "Hierarchical": lambda: get_clusters_h(_conn, vectors, n_clusters),
            "Hierarchical (Threshold)": lambda: get_clusters_h_threshold(_conn, vectors, distance_threshold)
        }
        if(not algorithm in switcher):
            raise Exception("Invalid algorithm")
        func = switcher.get(algorithm, lambda: None)
        labels, descriptions, centroids = func()
        return labels, descriptions, centroids

@st.cache_data(max_entries=4)
def get_clusters_kmeans(_conn, embeddings, n_clusters=10, random_state=42):
    # Connect to the database
    cursor = _conn.cursor()

    # Perform clustering
    n_init = 10
    kmeans = KMeans(n_clusters=n_clusters, n_init=n_init, random_state=random_state)
    kmeans.fit(embeddings)
    cluster_centers = kmeans.cluster_centers_

    # Normalize the cluster centers
    cluster_centers = cluster_centers / np.linalg.norm(cluster_centers, axis=1)[:, np.newaxis]

    # Lookup nearest 5 words for each normalized cluster center
    labels = []
    for i, center in enumerate(cluster_centers):
        closest_words = get_closest_words(center, cursor, k=5)
        labels.append(f"Cluster {i+1}: {', '.join(closest_words)}")

    cursor.close()

    return kmeans.labels_, labels, cluster_centers

@st.cache_data(max_entries=4)
def get_clusters_h(_conn, embeddings, n_clusters=10):
    # Connect to the database
    cursor = _conn.cursor()

    # Create a hierarchical clustering model with the specified number of clusters.
    # The 'affinity' is set to cosine to use cosine similarity as the distance metric.
    # The linkage criterion is set to 'average' which is the average of the distances.
    clustering = AgglomerativeClustering(n_clusters=n_clusters, metric='euclidean', linkage='ward')

    # Fit the model
    clustering.fit(embeddings)

    # Find the centers of the clusters
    cluster_centers = np.array([embeddings[clustering.labels_ == i].mean(axis=0) for i in range(n_clusters)])

    # Normalize the cluster centers
    cluster_centers = cluster_centers / np.linalg.norm(cluster_centers, axis=1)[:, np.newaxis]

    # Lookup nearest 5 words for each normalized cluster center
    labels = []
    for i, center in enumerate(cluster_centers):
        closest_words = get_closest_words(center, cursor, k=5)
        labels.append(f"Cluster {i+1}: {', '.join(closest_words)}")

    cursor.close()

    return clustering.labels_, labels, cluster_centers

@st.cache_data(max_entries=4)
def get_clusters_h_threshold(_conn, embeddings, distance_threshold=0.5):
    # Connect to the database
    cursor = _conn.cursor()

    # Create a hierarchical clustering model with no specified number of clusters.
    # Instead, use a distance threshold to determine the number of clusters.
    clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=distance_threshold, metric='cosine', linkage='complete')

    # Fit the model
    clustering.fit(embeddings)

    # The number of clusters is determined by the distance threshold
    n_clusters = clustering.n_clusters_

    if(n_clusters > 30):
        raise Exception(f"Too many clusters for the given threshold (30 max, {n_clusters} found).")

    # Find the centers of the clusters
    cluster_centers = np.array([embeddings[clustering.labels_ == i].mean(axis=0) for i in range(n_clusters)])

    # Normalize the cluster centers
    cluster_centers = cluster_centers / np.linalg.norm(cluster_centers, axis=1)[:, np.newaxis]

    # Lookup nearest 5 words for each normalized cluster center
    labels = []
    for i, center in enumerate(cluster_centers):
        closest_words = get_closest_words(center, cursor, k=5)
        labels.append(f"Cluster {i+1}: {', '.join(closest_words)}")

    cursor.close()

    return clustering.labels_, labels, cluster_centers
