import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

@st.cache_data(max_entries=4)
def get_tsne_data(embeddings, dimensions=2, random_state=42):

    # Perform t-SNE dimensionality reduction
    perplexity = max(1,min(25, len(embeddings) - 1))
    learning_rate = max(1, min(200, len(embeddings) // 10))

    tsne = TSNE(n_components=dimensions, perplexity=perplexity, metric="cosine", learning_rate=learning_rate, random_state=random_state)
    X_tsne = tsne.fit_transform(embeddings)

    return X_tsne

@st.cache_data(max_entries=4)
def render_tsne_plotly(xtsne, labelled_data, lines, label_descriptions, dimensions=2, height=1000):
    if dimensions not in [2, 3, 4]:
        raise ValueError("dimensions must be 2 or 3, or... 4")

    # Create the figure using Plotly Graph Objects
    fig = go.Figure()

    # Define a discrete color sequence
    if(len(label_descriptions) <= 10):
        colors = px.colors.qualitative.Plotly
    else:
        colors = px.colors.qualitative.Alphabet

    # Loop over the labels and add a scatter plot for each cluster
    for i, label_desc in enumerate(label_descriptions):
        # Extract the indices for the current label
        indices = [j for j, x in enumerate(labelled_data) if x == i]
        # Define the color for the current label using the modulo operator to cycle through colors
        color = colors[i % len(colors)]
        # Add a trace for the current label

        if dimensions == 2:
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
        else:
            fig.add_trace(
                go.Scatter3d(
                    x=xtsne[indices, 0],
                    y=xtsne[indices, 1],
                    z=xtsne[indices, 2],
                    mode="markers",
                    marker=dict(
                        size=10,
                        color=color,
                        opacity=0.7
                    ),
                    customdata=np.array(lines)[indices],
                    hovertemplate='%{customdata}<extra></extra>',
                    name=label_desc  # use the label description for the legend entry
                )
            )

    # Update layout and title
    fig.update_layout(
        autosize=False,
        # width=1000,
        height=height,
        #margin=dict(l=50, r=50, b=100, t=100, pad=4),
        legend=dict( 
            title='Cluster Descriptions',
            orientation="h",
            y=-0.15,  # position the legend below the plot
            xanchor="center",
            x=0.5
        )
    )

    if dimensions == 2:
        fig.update_layout(        
            xaxis=dict(showticklabels=False),
            yaxis=dict(showticklabels=False)
        )
    elif dimensions == 3:
        fig.update_layout(
            scene=dict(                
                xaxis=dict(showticklabels=False),
                yaxis=dict(showticklabels=False),
                zaxis=dict(showticklabels=False)
            )
        )

    return fig
