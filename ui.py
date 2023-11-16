import streamlit as st
from vectorclient import get_embeddings, reflect_vector
from gptclient import generate_cluster_names_many, name_clusters_array
from clusterclient import get_clusters
from vectordbclient import get_closest_words
from visualclient import get_tsne_data, render_tsne_plotly
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import psycopg2
import json

if 'tsne_data' not in st.session_state:
    st.session_state.tsne_data = None

if 'vectors' not in st.session_state:
    st.session_state.vectors = None

if 'labels' not in st.session_state:
    st.session_state.labels = None

if 'descriptions' not in st.session_state:
    st.session_state.descriptions = None

if 'centroids' not in st.session_state:
    st.session_state.centroids = None

if 'similarity_threshold' not in st.session_state:
    st.session_state.similarity_threshold = 0.18

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

if 'use3d' not in st.session_state:
    st.session_state.use3d = False

if 'randomSeed' not in st.session_state:
    st.session_state.randomSeed = 42

if 'clusteringAlgorithm' not in st.session_state:
    st.session_state.clusteringAlgorithm = "KMeans"

st.set_page_config(layout="wide")

def connectToDb():
    return psycopg2.connect(st.secrets["connectionString"])

conn = connectToDb()

col1, col2 = st.columns(2)

with col1:
    inputText = st.text_area("Enter your text items, separated by newlines.")
    isGenerate = st.button("Generate Scatter Plot")
    st.session_state.removeConceptText = st.text_input("Remove concept from data", help="If you have a concept that is common to all the items you can enter it here and then clustering will try to ignore that sentiment. For example, if you have a list of comments about 'car problems' you don't want the clustering to be dominated by the word 'car'. This is a feature that can really mess up the clustering if you enter text which isn't common to all text items because they will be modified as if they were which could take them literally anywhere in multidimentional vector space. When experimenting with this, try turninig off OpenAI cluster naming so you can see the underlying cluster concepts.")

    with st.expander("Automatic cluster identification", expanded=False):
        detectClusters = st.checkbox("Detect clusters automatically", value=True)
        st.session_state.clusteringAlgorithm = st.selectbox("Clustering algorithm", ["KMeans", "Hierarchical", "Hierarchical (Threshold)"], help="Hierarchical clustering may better results for data with broad categories. With a threshold, the algorithm choosing the number of clusters for you (tweakable via merging threshold slider).")
        if(not detectClusters or st.session_state.clusteringAlgorithm == "Hierarchical (Threshold)"):
            n_clusters = 1
        else:
            n_clusters = st.number_input("Specify number of clusters.", min_value=1, max_value=20, value=8, disabled=not detectClusters)
        if(st.session_state.clusteringAlgorithm == "Hierarchical (Threshold)"):
            distance_threshold = st.slider("Distance threshold", 0.0, 1.0, 0.31, 0.01, help="Increasing this makes items less likely to be merged, resulting in more clusters.")
            n_clusters = 1
        else: distance_threshold = None
        st.session_state.gptLabelling = st.checkbox("Use OpenAI to name clusters") and detectClusters        

    with st.expander("Filtering", expanded=False):
        filterChoice = st.selectbox("Filter type", ["Filter out", "Show filtered"], help="Show only (or filter out) items that are semantically similar to this text. This is useful if you want to focus on a particular theme or hide noise. Tip: a full stop '.' tends to match poor quality answers before good ones.")
        filterOut = filterChoice == "Filter out"
        st.session_state.comparison_text = st.text_input("Enter text for semantic filtering", help="Show only (or filter out) items that are semantically similar to this text. This is useful if you want to focus on a particular theme.")
        similarity_threshold = st.slider("Similarity threshold", 0.0, 1.0, 0.5, 0.01, help="Adjust this to show or hide more items that are similar to the comparison text.")
        filterOutChanged = filterOut != st.session_state.lastfilterOut
        st.session_state.lastfilterOut = filterOut
        similarity_changed = similarity_threshold != st.session_state.similarity_threshold
        st.session_state.similarity_threshold = similarity_threshold
        string_list = [s for s in inputText.strip().split('\n') if s.strip()]
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
        st.caption("If you want to name the clusters, click the 'Use OpenAI to name clusters' checkbox and click the 'Generate Scatter Plot' button again. This adds a few seconds to the processing time. Note that you can click on the items in the legend to hide or show that category.")

    with st.expander("Experimental", expanded=False):
        st.session_state.randomSeed = st.number_input("Random seed", min_value=0, value=42)
        st.caption("You can try navigating the data in 3d, but it won't make things easier. It's just for fun.")
        st.session_state.use3d = st.checkbox("Use 3D plot", False)
        if(st.session_state.centroids is not None and st.button("Show cluster vectors")):
            expressions = [f"!{desc} {json.dumps(list(st.session_state.centroids[i]))}" for i, desc in enumerate(st.session_state.descriptions)]
            st.text_area("Cluster vectors", "\n".join(expressions))

dimensions = 3 if st.session_state.use3d else 2

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
        fig = render_tsne_plotly(filtered_data, filtered_labels, filtered_string_list, st.session_state.descriptions, dimensions=dimensions)
        st.plotly_chart(fig)
        st.caption(f"Showing {len(filtered_data)} of {len(st.session_state.tsne_data)} items")

    if (isGenerate):
        st.session_state.vectors = get_embeddings(string_list, conn)
        if(st.session_state.removeConceptText.strip() != ""):
            vectorToRemove = get_embeddings([st.session_state.removeConceptText.strip()], conn)[0]
            st.session_state.vectors = np.apply_along_axis(reflect_vector, 1, st.session_state.vectors, vectorToRemove)
        labels, descriptions, centroids = get_clusters(conn,
                                                       st.session_state.clusteringAlgorithm,
                                                       st.session_state.vectors,
                                                       n_clusters=n_clusters,
                                                       random_state=st.session_state.randomSeed,
                                                       distance_threshold=distance_threshold)
        st.session_state.labels = labels
        st.session_state.descriptions = descriptions
        st.session_state.centroids = centroids
        if(st.session_state.gptLabelling and len(st.session_state.descriptions) > 0 and len(st.session_state.descriptions) < 25):
            tasks = []
            for(i, label) in enumerate(st.session_state.descriptions):
                samples = np.array(string_list)[st.session_state.labels == i]
                sample_count = min(20, len(samples))
                samples = np.random.choice(samples, sample_count, replace=False)
                tasks.append({"labels": label, "samples": samples})
            st.session_state.descriptions = generate_cluster_names_many(tasks)

        st.session_state.tsne_data = get_tsne_data(st.session_state.vectors, dimensions=dimensions, random_state=st.session_state.randomSeed)
        fig = render_tsne_plotly(st.session_state.tsne_data, st.session_state.labels, string_list, st.session_state.descriptions, dimensions=dimensions)
        st.plotly_chart(fig)
    elif(st.session_state.tsne_data is not None and not isfilter):
        tsneDim = st.session_state.tsne_data.shape[1]
        if tsneDim != dimensions:
            st.session_state.tsne_data = get_tsne_data(st.session_state.vectors, dimensions=dimensions, random_state=st.session_state.randomSeed)
        fig = render_tsne_plotly(st.session_state.tsne_data, st.session_state.labels, string_list, st.session_state.descriptions, dimensions=dimensions)
        st.plotly_chart(fig)

conn.close()