import psycopg2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import plotly.graph_objects as go

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

def render_tsne_plotly(xtsne, labels, lines):
    # Create the figure using Plotly Graph Objects
    fig = go.Figure()

    # Add scatter plot
    fig.add_trace(
        go.Scatter(
            x=xtsne[:, 0],
            y=xtsne[:, 1],
            mode="markers",
            marker=dict(
                size=10,
                color=labels,
                colorscale='Jet',
                showscale=False
            ),
            customdata=lines,
            hovertemplate='%{customdata}<extra></extra>',
            name=''
        )
    )

    # Update layout and title
    fig.update_layout(
        title='t-SNE visualization',
        autosize=False,
        width=800,
        height=800,
        margin=dict(l=50, r=50, b=100, t=100, pad=4),
        coloraxis_colorbar=dict(
            title='Cluster'
        ),
        xaxis=dict(showticklabels=False),
        yaxis=dict(showticklabels=False)
    )

    return fig