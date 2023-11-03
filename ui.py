import streamlit as st
from adaclient import get_embeddings
from gptclient import generate_cluster_name, generate_cluster_names_many
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

if 'descriptions' not in st.session_state:
    st.session_state.descriptions = None

if 'similarity_threshold' not in st.session_state:
    st.session_state.similarity_threshold = 0.5

if 'comparison_text' not in st.session_state:
    st.session_state.comparison_text = ""

if 'lastfilterOut' not in st.session_state:
    st.session_state.lastfilterOut = False

if 'gptLabelling' not in st.session_state:
    st.session_state.gptLabelling = False

conn = psycopg2.connect(st.secrets["connectionString"])

st.set_page_config(layout="wide")

st.title("Semantic Clustering")

col1, col2 = st.columns(2)

with col1:
    inputText = st.text_area("Enter your text items, separated by newlines.")
    n_clusters = st.number_input("Specify number of clusters", min_value=1, max_value=20, value=8)
    st.session_state.gptLabelling = st.checkbox("Use GPT to label clusters")
    isGenerate = st.button("Generate Scatter Plot")

    with st.expander("Filtering", expanded=False):
        st.caption("Show or hide items that are similar to a given text.")
        st.session_state.comparison_text = st.text_input("Enter text for semantic filtering")
        similarity_threshold = st.slider("Similarity threshold", 0.0, 1.0, 0.5, 0.01)
        filterOut = st.checkbox("Filter out")
        filterOutChanged = filterOut != st.session_state.lastfilterOut
        st.session_state.lastfilterOut = filterOut
        similarity_changed = similarity_threshold != st.session_state.similarity_threshold
        st.session_state.similarity_threshold = similarity_threshold
        string_list = inputText.strip().split('\n')
        isfilter = (st.button("Filter") or similarity_changed or filterOutChanged) \
            and st.session_state.comparison_text.strip() != "" \
            and st.session_state.tsne_data is not None

with col2:
    if (isfilter):
        comparison_embedding = get_embeddings([st.session_state.comparison_text.strip()], conn)[0]
        similarities = cosine_similarity(st.session_state.vectors, comparison_embedding.reshape(1, -1)).flatten()
        if filterOut:
            mask = similarities < (1 - st.session_state.similarity_threshold)
        else:
            mask = similarities >= (1-st.session_state.similarity_threshold)
        filtered_data = st.session_state.tsne_data[mask]
        filtered_labels = st.session_state.labels[mask]
        filtered_string_list = np.array(string_list)[mask]
        fig = render_tsne_plotly(filtered_data, filtered_labels, filtered_string_list, st.session_state.descriptions)
        st.plotly_chart(fig)
        st.caption(f"Showing {len(filtered_data)} of {len(st.session_state.tsne_data)} items")
        
    if (isGenerate):
        st.session_state.vectors = get_embeddings(string_list, conn)
        st.session_state.labels, st.session_state.descriptions = get_clusters(conn, st.session_state.vectors, n_clusters)
        if(st.session_state.gptLabelling):
            tasks = []
            for(i, label) in enumerate(st.session_state.descriptions):
                samples = np.array(string_list)[st.session_state.labels == i]
                sample_count = min(20, len(samples))
                samples = np.random.choice(samples, sample_count, replace=False)
                tasks.append({"labels": label, "samples": samples})
            st.session_state.descriptions = generate_cluster_names_many(tasks)
            #st.session_state.descriptions[i] = (generate_cluster_name(label, samples))

        st.session_state.tsne_data, _ = get_tsne_data(st.session_state.vectors, n_clusters)
        fig = render_tsne_plotly(st.session_state.tsne_data, st.session_state.labels, string_list, st.session_state.descriptions)
        st.plotly_chart(fig)
    elif(st.session_state.tsne_data is not None and not isfilter):
        fig = render_tsne_plotly(st.session_state.tsne_data, st.session_state.labels, string_list, st.session_state.descriptions)
        st.plotly_chart(fig)