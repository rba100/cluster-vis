import streamlit as st
from adaclient import get_embeddings
from gptclient import generate_cluster_name, generate_cluster_names_many
from clusterclient import get_clusters
from vectordbclient import get_closest_words, reflect_across_vector
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

if 'filterMask' not in st.session_state:
    st.session_state.filterMask = None

if 'removeConceptText' not in st.session_state:
    st.session_state.removeConceptText = ""

conn = psycopg2.connect(st.secrets["connectionString"])

st.set_page_config(layout="wide")

st.title("Semantic Clustering")

col1, col2 = st.columns(2)

with col1:
    inputText = st.text_area("Enter your text items, separated by newlines.")
    isGenerate = st.button("Generate Scatter Plot")

    with st.expander("Automatic cluster identification", expanded=False):
        detectClusters = st.checkbox("Detect clusters automatically", value=True)
        if(not detectClusters):
            n_clusters = 1
        else:
            n_clusters = st.number_input("Specify number of clusters.", min_value=1, max_value=20, value=8, disabled=not detectClusters)
        st.session_state.gptLabelling = st.checkbox("Use OpenAI to name clusters") and detectClusters
        st.session_state.removeConceptText = st.text_input("Remove concept from clustering")

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
        isfilter = (st.button("Apply filter") or similarity_changed or filterOutChanged) \
            and st.session_state.comparison_text.strip() != "" \
            and st.session_state.tsne_data is not None
        showFilteredTextItems = st.session_state.filterMask is not None and st.button("Show filtered text items")
        if showFilteredTextItems:
            st.text_area("Filtered text items", "\n".join(np.array(string_list)[st.session_state.filterMask]))


    with st.expander("Help", expanded=False,):
        st.caption("This tool groups similar text items together and presents them as a visual plot. You can then mouse over the points to see the corresponding text and manually identify common themes.")
        st.caption("The first step is to enter your text items in the box above, one per line, then click the 'Generate Scatter Plot'.")
        st.caption("To help you spot common themes you can specify a number of clusters to identiy and these will be colour-coded. This feature uses traditional statistical methods to identify clusters of similar items.")
        st.caption("If you want to filter out items that are similar to a given text, enter the text in the box in the filter section and click the 'Filter' button. You can also choose to filter out or filter in the items.")
        st.caption("If you want to name the clusters, click the 'Use OpenAI to name clusters' checkbox and click the 'Generate Scatter Plot' button again. This adds a few seconds to the processing time. Note that you can click on the items in the legend to hide or show that category.")
        st.caption("'Remove concept from clustering'. If you have a concept that is common to all the items you can enter it here and then clustering will try to ignore that sentiment. For example, if you have a list of comments about 'car problems' you don't want the clustering to be dominated by the word 'car'. This is a feature that can really mess up the clustering if you enter text which isn't common to all text items because they will be modified as if they were which could take them literally anywhere in multidimentional vector space. When experimenting with this, try turninig off OpenAI cluster naming so you can see the underlying cluster concepts.")

with col2:
    if (isfilter):
        comparison_embedding = get_embeddings([st.session_state.comparison_text.strip()], conn)[0]
        similarities = cosine_similarity(st.session_state.vectors, comparison_embedding.reshape(1, -1)).flatten()
        if filterOut:
            st.session_state.filterMask = similarities < (1 - st.session_state.similarity_threshold)
        else:
            st.session_state.filterMask = similarities >= (1-st.session_state.similarity_threshold)
        filtered_data = st.session_state.tsne_data[st.session_state.filterMask]
        filtered_labels = st.session_state.labels[st.session_state.filterMask]
        filtered_string_list = np.array(string_list)[st.session_state.filterMask]
        fig = render_tsne_plotly(filtered_data, filtered_labels, filtered_string_list, st.session_state.descriptions)
        st.plotly_chart(fig)
        st.caption(f"Showing {len(filtered_data)} of {len(st.session_state.tsne_data)} items")
        
    if (isGenerate):
        st.session_state.vectors = get_embeddings(string_list, conn)
        if(st.session_state.removeConceptText.strip() != ""):
            vectorToRemove = get_embeddings([st.session_state.removeConceptText.strip()], conn)[0]
            st.session_state.vectors = np.apply_along_axis(reflect_across_vector, 1, st.session_state.vectors, vectorToRemove)
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