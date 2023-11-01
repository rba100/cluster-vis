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
from adaclient import get_embeddings
from vectordbclient import get_closest_words

data = pd.read_csv("argos-1.tsv", sep='\t')
data = data[data["QB2"].apply(lambda x: isinstance(x, str) and x.lower().strip() != "nothing")]
lines = data['QB2'].tolist()

# Connect to the database
conn = psycopg2.connect(host='localhost', database='postgres', user='postgres', password='postgres')
cursor = conn.cursor()

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

# Lookup nearest 5 words for each normalized cluster center
labels = []
for i, center in enumerate(cluster_centers):
    closest_words = get_closest_words(center, cursor)
    labels.append(f"Cluster {i+1}: {', '.join(closest_words)}")

# Close the connection
conn.close()

# Perform t-SNE dimensionality reduction
tsne = TSNE(n_components=2)
X_tsne = tsne.fit_transform(embeddings)


# chart colors
colors = data['QB1'].tolist()
colorLabels, unique = pd.factorize(colors)
colormap = mcolors.ListedColormap(plt.cm.jet(np.linspace(0, 1, len(unique))))

# Plot the t-SNE reduced data
plt.figure(figsize=(8, 8))
#scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=kmeans.labels_, cmap='jet', alpha=0.6)
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=colorLabels, cmap='jet', alpha=0.6)


#wrapped_labels = [textwrap.fill(label, width=30) for label in labels]
#legend1 = plt.legend(handles=scatter.legend_elements(num=n_clusters)[0], title="Clusters", labels=wrapped_labels, loc='best', bbox_to_anchor=(1, 1))



legend_handles = [plt.Line2D([0], [0], marker='o', color='w', label=unique[i], markersize=10, markerfacecolor=colormap(i)) for i in range(len(unique))]
plt.legend(handles=legend_handles, title="QB2 Values", loc='best', bbox_to_anchor=(1, 1))
#legend_labels = [f"{label}" for label in unique]
#legend1 = plt.legend(handles=scatter.legend_elements(num=len(unique))[0], title="QB1 Values", labels=legend_labels, loc='best', bbox_to_anchor=(1, 1))
plt.title('t-SNE visualization')
#plt.xlabel('t-SNE feature 1')
#plt.ylabel('t-SNE feature 2')
#plt.tight_layout()
#plt.savefig('chart.png', dpi=300)

# Add mouse-over tooltips using mplcursors
mplcursors.cursor(hover=True).connect(
    "add", lambda sel: sel.annotation.set_text(lines[sel.index])
)

plt.show()
