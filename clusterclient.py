import numpy as np
from sklearn.cluster import KMeans
from vectordbclient import get_closest_words;

def get_clusters(conn, embeddings, n_clusters=10):
    # Connect to the database
    cursor = conn.cursor()

    # Perform clustering
    n_init = 10
    kmeans = KMeans(n_clusters=n_clusters, n_init=n_init, random_state=42)
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