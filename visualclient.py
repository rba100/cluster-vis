import psycopg2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import plotly.graph_objects as go
import plotly.express as px

def get_tsne_data(embeddings, n_clusters=10):

    # Perform clustering
    n_init = 10
    kmeans = KMeans(n_clusters=n_clusters, n_init=n_init)
    kmeans.fit(embeddings)
    cluster_centers = kmeans.cluster_centers_

    # Normalize the cluster centers
    cluster_centers = cluster_centers / np.linalg.norm(cluster_centers, axis=1)[:, np.newaxis]

    # Perform t-SNE dimensionality reduction
    perplexity = min(25, len(embeddings) - 1)
    learning_rate = max(1, min(200, len(embeddings) // 10))

    tsne = TSNE(n_components=2, perplexity=perplexity, metric="cosine", learning_rate=learning_rate)
    X_tsne = tsne.fit_transform(embeddings)

    return X_tsne, kmeans.labels_

def render_tsne_plotly(xtsne, labels, lines, label_descriptions):
    # Create the figure using Plotly Graph Objects
    fig = go.Figure()

    # Define a discrete color sequence
    colors = px.colors.qualitative.Plotly

    # Loop over the labels and add a scatter plot for each cluster
    for i, label_desc in enumerate(label_descriptions):
        # Extract the indices for the current label
        indices = [j for j, x in enumerate(labels) if x == i]
        # Define the color for the current label using the modulo operator to cycle through colors
        color = colors[i % len(colors)]
        # Add a trace for the current label
        fig.add_trace(
            go.Scatter(
                x=xtsne[indices, 0],
                y=xtsne[indices, 1],
                mode="markers",
                marker=dict(
                    size=10,
                    color=color,  # use the discrete color for the current label
                    opacity=0.9
                ),
                customdata=np.array(lines)[indices],
                hovertemplate='%{customdata}<extra></extra>',
                name=label_desc  # use the label description for the legend entry
            )
        )

    # Update layout and title
    fig.update_layout(
        title='t-SNE visualization',
        autosize=False,
        width=800,
        height=1000,
        margin=dict(l=50, r=50, b=100, t=100, pad=4),
        legend=dict(
            title='Cluster Descriptions',
            orientation="h",
            y=-0.15,  # position the legend below the plot
            xanchor="center",
            x=0.5
        ),
        xaxis=dict(showticklabels=False),
        yaxis=dict(showticklabels=False)
    )

    return fig
