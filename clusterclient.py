import psycopg2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import plotly.graph_objects as go
from gptclient import generate_cluster_name
from adaclient import get_embeddings
from vectordbclient import get_closest_words;

def get_clusters(lines, n_clusters=10):
    # Connect to the database
    conn = psycopg2.connect(host='localhost', database='postgres', user='postgres', password='postgres')
    cursor = conn.cursor()

    # Get embeddings for all lines at once
    embeddings = np.array(get_embeddings(lines, conn))

    # Perform clustering
    n_init = 10
    kmeans = KMeans(n_clusters=n_clusters, n_init=n_init)
    kmeans.fit(embeddings)
    cluster_centers = kmeans.cluster_centers_

    # Normalize the cluster centers
    cluster_centers = cluster_centers / np.linalg.norm(cluster_centers, axis=1)[:, np.newaxis]

    # Lookup nearest 5 words for each normalized cluster center
    labels = []
    for i, center in enumerate(cluster_centers):
        closest_words = get_closest_words(center, cursor)
        labels.append(f"Cluster {i+1}: {', '.join(closest_words)}")

    # Close the connection
    conn.close()

    return embeddings, kmeans.labels_, labels