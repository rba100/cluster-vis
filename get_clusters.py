import psycopg2
import openai
import numpy as np
from sklearn.cluster import KMeans
from adaclient import get_embeddings
from gptclient import name_clusters
from scipy.stats import chi2_contingency

def get_closest_words(embedding, cursor):
    embedding_str = ','.join(map(str, embedding))
    embedding_str = f'[{embedding_str}]'
    query = """
    SELECT word
    FROM words -- WHERE isCommon = true
    ORDER BY embedding <=> %s
    LIMIT 5
    """
    cursor.execute(query, (embedding_str,))
    results = cursor.fetchall()
    return [result[0] for result in results]

# Connect to the database
conn = psycopg2.connect(host='localhost', database='postgres', user='postgres', password='postgres')
cursor = conn.cursor()

# Read lines from input.txt and strip newlines
with open("input.txt", "r") as f:
    lines = [line.strip() for line in f.readlines()]

# Get embeddings for all lines at once
embeddings = np.array(get_embeddings(lines, conn))

# Perform clustering
n_clusters = 10
n_init = 10
kmeans = KMeans(n_clusters=n_clusters, n_init=n_init)
kmeans.fit(embeddings)
cluster_centers = kmeans.cluster_centers_

# Normalize the cluster centers
cluster_centers = cluster_centers / np.linalg.norm(cluster_centers, axis=1)[:, np.newaxis]

stringbuilder = ""

# Lookup nearest 5 words for each normalized cluster center
for i, center in enumerate(cluster_centers):
    closest_words = get_closest_words(center, cursor)
    print(f"Cluster {i+1} closest words: {', '.join(closest_words)}")
    stringbuilder = stringbuilder + f"Cluster {i+1} closest words: {', '.join(closest_words)}\n"

# Use GPT-4 to name Clusters
summary = name_clusters(stringbuilder)
print(summary)

# Close the connection
conn.close()
