import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from vectordbclient import get_closest_words;

def get_clusters(conn, embeddings, n_clusters=10, random_state=42):
    # Connect to the database
    cursor = conn.cursor()

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

    return kmeans.labels_, labels

def get_clusters_h(conn, embeddings, n_clusters=10, random_state=0):
    # Connect to the database
    cursor = conn.cursor()

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

    return clustering.labels_, labels