import psycopg2
import openai
import numpy as np
from sklearn.cluster import KMeans
from gptclient import generate_cluster_name
from adaclient import get_embeddings
from vectordbclient import get_closest_words;

# Read lines from input.txt and strip newlines
with open("input.txt", "r") as f:
    lines = [line.strip() for line in f.readlines()]

# Get embeddings for all lines at once
conn = psycopg2.connect(host='localhost', database='postgres', user='postgres', password='postgres')
cursor = conn.cursor()
embeddings = np.array(get_embeddings(lines, conn))
words =  [{"line": line, "embedding": vector} for line, vector in zip(lines, embeddings)]

# Perform clustering
n_clusters = 10
n_init = 10
kmeans = KMeans(n_clusters=n_clusters, n_init=n_init)
clusters = kmeans.fit(embeddings)
cluster_centers = kmeans.cluster_centers_
# Normalize the cluster centers
cluster_centers = cluster_centers / np.linalg.norm(cluster_centers, axis=1)[:, np.newaxis]

cluster_words = {}
# Lookup nearest 5 words for each normalized cluster center
for i, center in enumerate(cluster_centers):
    closest_words = get_closest_words(center, cursor)
    cluster_words[i] = closest_words
    print(f"Cluster {i+1} closest words: {', '.join(closest_words)}")

# get some examples from each cluster for GPT
cluster_samples = {}
for cluster_num in range(n_clusters):
    cluster_indices = np.where(kmeans.labels_ == cluster_num)[0]
    sample_indices = np.random.choice(cluster_indices, size=min(20, len(cluster_indices)), replace=False)
    samples = [words[index]['line'] for index in sample_indices]
    cluster_samples[cluster_num] = samples

cluster_names = []
for cluster_num in n_clusters:
    name = generate_cluster_name(cluster_words[cluster_num], cluster_samples[cluster_num])
    cluster_names.append(name)

# Close the connection
conn.close()
