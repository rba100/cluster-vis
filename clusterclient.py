import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_samples
from scipy.stats import iqr
from kneed import KneeLocator
from vectordbclient import get_closest_words;
import streamlit as st

def get_clusters(_conn, algorithm, vectors, n_clusters, random_state=None, distance_threshold=None):
    switcher = {
        "KMeans": lambda: get_clusters_kmeans(_conn, vectors, n_clusters, random_state=random_state),
        "KMeans (Elbow)": lambda: get_optimal_clusters_kmeans_elbow(_conn, vectors),
        "KMeans (Silhouette)": lambda: get_optimal_clusters_kmeans_silhouette2(_conn, vectors),
        "Hierarchical": lambda: get_clusters_h(_conn, vectors, n_clusters),
        "Hierarchical (Threshold)": lambda: get_clusters_h_threshold(_conn, vectors, distance_threshold)
    }
    if(not algorithm in switcher):
        raise Exception("Invalid algorithm. Choices: KMeans, Hierarchical, Hierarchical (Threshold)")
    func = switcher.get(algorithm, lambda: None)
    labels, descriptions, centroids = func()
    return labels, descriptions, centroids

@st.cache_data(max_entries=4, show_spinner="Clustering...")
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

@st.cache_data(max_entries=4, show_spinner="Clustering...")
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

@st.cache_data(max_entries=4, show_spinner="Clustering...")
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

@st.cache_data(max_entries=1, show_spinner="Clustering...")
def get_optimal_clusters_kmeans_elbow(_conn, embeddings, random_state=42):
    # Connect to the database
    cursor = _conn.cursor()

    maxClusters = min(30, len(embeddings) - 1)

    wcss = []
    for i in range(1, maxClusters):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=random_state)
        kmeans.fit(embeddings)
        wcss.append(kmeans.inertia_)

    # Use the elbow method to find the optimal number of clusters
    kn = KneeLocator(range(1,maxClusters), wcss, curve='convex', direction='decreasing')
    n_clusters = kn.elbow

    # Perform clustering with the optimal number of clusters
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=300, n_init=10, random_state=random_state)
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

@st.cache_data(max_entries=1, show_spinner="Clustering...")
def get_optimal_clusters_kmeans_silhouette(_conn, embeddings):
    # Connect to the database
    cursor = _conn.cursor()

    maxClusters = min(30, len(embeddings) -1)

    silhouette_scores = []
    for i in range(2, maxClusters):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(embeddings)
        silhouette_scores.append(silhouette_score(embeddings, kmeans.labels_))

    # Find the number of clusters that gives the highest silhouette score
    n_clusters = silhouette_scores.index(max(silhouette_scores)) + 2

    # Perform clustering with the optimal number of clusters
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
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

@st.cache_data(max_entries=1, show_spinner="Clustering...")
def get_optimal_clusters_kmeans_silhouette2(_conn, embeddings, random_state=42):
    # Connect to the database
    cursor = _conn.cursor()

    maxClusters = min(30, len(embeddings) -1)

    silhouette_scores = []
    valid_cluster_numbers = []
    for i in range(2, maxClusters):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=random_state)
        kmeans.fit(embeddings)
        labels = kmeans.labels_
        avg_silhouette_score = silhouette_score(embeddings, labels)
        silhouette_scores.append(avg_silhouette_score)

        # Check if all clusters have a silhouette score greater than the average score
        individual_silhouette_scores = silhouette_samples(embeddings, labels)
        if all(individual_silhouette_score >= avg_silhouette_score for individual_silhouette_score in individual_silhouette_scores):
            # Check if the sizes of the clusters are not too different
            cluster_sizes = np.bincount(labels)
            if iqr(cluster_sizes) <= np.median(cluster_sizes):
                valid_cluster_numbers.append(i)

    # Find the number of clusters among the valid ones that gives the highest silhouette score
    valid_silhouette_scores = [silhouette_scores[i-2] for i in valid_cluster_numbers] if len(valid_cluster_numbers) > 0 else silhouette_scores
    n_clusters = valid_cluster_numbers[valid_silhouette_scores.index(max(valid_silhouette_scores))] if len(valid_cluster_numbers) > 0 else silhouette_scores.index(max(silhouette_scores)) + 2

    # Perform clustering with the optimal number of clusters
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=300, n_init=10, random_state=random_state)
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