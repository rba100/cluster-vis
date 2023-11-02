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

conn = psycopg2.connect(host='localhost', database='postgres', user='postgres', password='postgres')
cursor = conn.cursor()

#st.set_page_config(layout="wide")

# Streamlit app starts here
st.title("Vector Embeddings and Clustering")


block_of_text = st.text_area("Enter your block of text, separated by newlines")
n_clusters = st.number_input("Specify number of clusters", min_value=1, max_value=20, value=3)
similarity_threshold = st.slider("Similarity threshold", 0.0, 1.0, 0.5, 0.01)
filterOut = st.checkbox("Filter out")
filterOutChanged = filterOut != st.session_state.lastfilterOut
st.session_state.lastfilterOut = filterOut
similarity_changed = similarity_threshold != st.session_state.similarity_threshold
st.session_state.similarity_threshold = similarity_threshold
st.session_state.comparison_text = st.text_input("Enter text to compare")
string_list = block_of_text.strip().split('\n')
isfilter = st.button("Filter") or similarity_changed or filterOutChanged
isGenerate = st.button("Generate Scatter Plot")

    
if (isfilter or (similarity_changed and st.session_state.tsne_data is not None and st.session_state.comparison_text.strip() != "")):
    comparison_embedding = np.array(get_embeddings([st.session_state.comparison_text.strip()], conn)[0])
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
    
if (isGenerate):
    st.session_state.vectors, st.session_state.labels, descriptions = get_clusters(string_list, n_clusters)
    st.session_state.tsne_data, _ = get_tsne_data(string_list, n_clusters)
    fig = render_tsne_plotly(st.session_state.tsne_data, st.session_state.labels, string_list)
    st.plotly_chart(fig)
elif(st.session_state.tsne_data is not None and not isfilter):
    fig = render_tsne_plotly(st.session_state.tsne_data, st.session_state.labels, string_list)
    st.plotly_chart(fig)