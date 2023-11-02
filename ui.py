import streamlit as st
from adaclient import get_embeddings
from clusterclient import get_clusters
from vectordbclient import get_closest_words
from visualclient import get_tsne_data, render_tsne_plotly
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import psycopg2

if 'tsne_data' not in st.session_state:
    st.session_state.tsne_data = None

if 'vectors' not in st.session_state:
    st.session_state.vectors = None

if 'labels' not in st.session_state:
    st.session_state.labels = None

if 'similarity_threshold' not in st.session_state:
    st.session_state.similarity_threshold = 0.5

if 'comparison_text' not in st.session_state:
    st.session_state.comparison_text = ""

if 'lastfilterOut' not in st.session_state:
    st.session_state.lastfilterOut = False

conn = psycopg2.connect(st.secrets["connectionString"])

st.set_page_config(layout="wide")

st.title("Semantic Clustering")

col1, col2 = st.columns(2)

with col1:
    inputText = st.text_area("Enter your text items, separated by newlines.")
    n_clusters = st.number_input("Specify number of clusters", min_value=1, max_value=20, value=8)
    isGenerate = st.button("Generate Scatter Plot")

    st.subheader("Semantic Filtering")
    similarity_threshold = st.slider("Similarity threshold", 0.0, 1.0, 0.5, 0.01)
    filterOut = st.checkbox("Filter out")
    filterOutChanged = filterOut != st.session_state.lastfilterOut
    st.session_state.lastfilterOut = filterOut
    similarity_changed = similarity_threshold != st.session_state.similarity_threshold
    st.session_state.similarity_threshold = similarity_threshold
    st.session_state.comparison_text = st.text_input("Enter text for semantic filtering")
    string_list = inputText.strip().split('\n')
    st.caption("Items with similarity below the threshold will be hidden")
    isfilter = (st.button("Filter") and st.session_state.comparison_text != '') or similarity_changed or filterOutChanged

with col2:
    if (isfilter or (similarity_changed and st.session_state.tsne_data is not None and st.session_state.comparison_text.strip() != "")):
        comparison_embedding = get_embeddings([st.session_state.comparison_text.strip()], conn)[0]
        similarities = cosine_similarity(st.session_state.vectors, comparison_embedding.reshape(1, -1)).flatten()
        if filterOut:
            mask = similarities < (1 - st.session_state.similarity_threshold)
        else:
            mask = similarities >= (1-st.session_state.similarity_threshold)
        filtered_data = st.session_state.tsne_data[mask]
        filtered_labels = st.session_state.labels[mask]
        filtered_string_list = np.array(string_list)[mask]
        fig = render_tsne_plotly(filtered_data, filtered_labels, filtered_string_list)
        st.plotly_chart(fig)
        st.caption(f"Showing {len(filtered_data)} of {len(st.session_state.tsne_data)} items")
        
    if (isGenerate):
        st.session_state.vectors = get_embeddings(string_list, conn)
        st.session_state.labels, descriptions = get_clusters(conn, st.session_state.vectors, n_clusters)
        st.session_state.tsne_data, _ = get_tsne_data(st.session_state.vectors, n_clusters)
        fig = render_tsne_plotly(st.session_state.tsne_data, st.session_state.labels, string_list).
        st.plotly_chart(fig)
    elif(st.session_state.tsne_data is not None and not isfilter):
        fig = render_tsne_plotly(st.session_state.tsne_data, st.session_state.labels, string_list)
        st.plotly_chart(fig)