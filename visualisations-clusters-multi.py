import psycopg2
import openai
import pandas as pd
import textwrap
import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import mplcursors
from gptclient import generate_cluster_name
from adaclient import get_embeddings
from vectordbclient import get_closest_words;

# Read lines from input.txt and strip newlines
#with open("input.txt", "r") as f:
#    lines = [line.strip() for line in f.readlines()]

free_text_cols = ["QC2_2","QC2_3","QC2_4","QC2_5","QC2_6","QC2_7","QC2_8","QC2_9","QC2_10","QC2_11","QC2_12","QC2_13","QC2_14","QC2_15","QC2_16","QC2_17","QC2_18","QC2_19","QC2_20","QC2_21","QC2_22","QC2_23","QC2_24","QC2_25","QC2_26","QC2_27","QC2_28","QC2_29","QC2_30","QC2_31","QC2_32"]
#qc1FilterBad = ["Highly dissatisfied", "Dissatisfied", "Neither satisfied nor dissatisfied"]
qc1FilterBad = ["Highly dissatisfied", "Dissatisfied"] 
qc1FilterGood = ["Highly satisfied", "Satisfied"]

data = pd.read_csv("argos-1.tsv", sep='\t', dtype=str, na_filter=False)

lines = []

for col in free_text_cols:
    ratingCol = col[:2] + "1" + col[3:]
    extraDataCol = col[:2] + "3" + col[3:]
    ratings = data[ratingCol]
    bitmapFilter = data[ratingCol].apply(lambda x: isinstance(x, str) and x in qc1FilterGood)
    vals = data[bitmapFilter][col].tolist()
    extraVals = data[bitmapFilter][extraDataCol].tolist()
    combined = [f"{vals[i]} ({extraVals[i]})" for i in range(len(vals))]
    lines.extend([str(v).strip() for v in combined if str(v).strip()])  # Add non-empty strings to the list

# Connect to the database
conn = psycopg2.connect(host='localhost', database='postgres', user='postgres', password='postgres')
cursor = conn.cursor()

# Get embeddings for all lines at once
embeddings = get_embeddings(lines, conn)
words =  [{"line": line, "embedding": vector} for line, vector in zip(lines, embeddings)]

# Perform clustering
n_clusters = 8
kmeans = KMeans(n_clusters=n_clusters, n_init="auto", init="k-means++")
kmeans.fit(embeddings)
cluster_centers = kmeans.cluster_centers_

# Normalize the cluster centers
cluster_centers = cluster_centers / np.linalg.norm(cluster_centers, axis=1)[:, np.newaxis]

# Lookup nearest 5 words for each normalized cluster center
labels = []
cluster_words_arrays = {}
for i, center in enumerate(cluster_centers):
    closest_words = get_closest_words(center, cursor)
    cluster_words_arrays[i] = closest_words
    labels.append(f"Cluster {i+1}: {', '.join(closest_words)}")

# get some examples from each cluster for GPT
cluster_samples = {}
for cluster_num in range(n_clusters):
    cluster_indices = np.where(kmeans.labels_ == cluster_num)[0]
    sample_indices = np.random.choice(cluster_indices, size=min(20, len(cluster_indices)), replace=False)
    samples = [words[index]['line'] for index in sample_indices]
    cluster_samples[cluster_num] = samples

cluster_names = []
for cluster_num in range(n_clusters):
    #name = generate_cluster_name(cluster_words_arrays[cluster_num], cluster_samples[cluster_num])
    name = ", ".join(cluster_words_arrays[cluster_num])
    cluster_names.append(name)

# Close the connection
conn.close()

# Perform t-SNE dimensionality reduction
tsne = TSNE(n_components=2, init='pca', random_state=42, perplexity=20)
X_tsne = tsne.fit_transform(embeddings)

# Plot the t-SNE reduced data
plt.figure(figsize=(8, 8))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=kmeans.labels_, cmap='jet', alpha=0.6)


wrapped_labels = [textwrap.fill(label, width=20) for label in cluster_names]
legend1 = plt.legend(handles=scatter.legend_elements(num=n_clusters)[0], title="Clusters", labels=wrapped_labels, loc='upper left', bbox_to_anchor=(1, 1))

plt.title('t-SNE visualization')
#plt.xlabel('t-SNE feature 1')
#plt.ylabel('t-SNE feature 2')
#plt.tight_layout()
#plt.savefig('chart.png', dpi=300)

# Add mouse-over tooltips using mplcursors
mplcursors.cursor(hover=True).connect(
    "add", lambda sel: sel.annotation.set_text(lines[sel.index])
)

#plt.tight_layout()
plt.show()
