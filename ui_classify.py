import streamlit as st
from vectorclient import get_embeddings, get_embeddings_exp, getFieldName
from gptclient import generate_cluster_names_many
from clusterclient import get_clusters, get_clusters_h
from vectordbclient import get_closest_words
from visualclient import get_tsne_data, render_tsne_plotly
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import psycopg2

def main():

    if 'data_strings_raw' not in st.session_state:
        st.session_state.data_strings_raw = ''

    if 'labels_strings_raw' not in st.session_state:
        st.session_state.labels_strings_raw = ''

    if 'data_strings' not in st.session_state:
        st.session_state.data_strings = None

    if 'labels_strings' not in st.session_state:
        st.session_state.labels_strings = None

    if 'labels_thresholds' not in st.session_state:
        st.session_state.labels_thresholds = {}

    if 'data_vectors' not in st.session_state:
        st.session_state.data_vectors = None

    if 'labels_vectors' not in st.session_state:
        st.session_state.labels_vectors = None

    if 'dataframe' not in st.session_state:
        st.session_state.dataframe = None

    if 'similarity' not in st.session_state:
        st.session_state.similarity = None

    def connectToDb():
        return psycopg2.connect(st.secrets["connectionString"])

    conn = connectToDb()

    tab1, tab2, tab3 = st.tabs(["Configure", "Tabulate", "Tune"])

    with tab1:
        st.session_state.data_strings_raw = st.text_area("Enter your text items", value=st.session_state.data_strings_raw)
        st.session_state.labels_strings_raw = st.text_area("Enter your labels", st.session_state.labels_strings_raw)
        submitText = "Submit" if st.session_state.data_vectors is None else "Apply changes"
        apply = st.button(submitText)

        st.subheader("Instructions")
        st.caption("Your labels will be sematically matched to the data. You can see how a label matches to the data by adjusting the threshold in the 'Tune' tab. The threshold is the minimum similarity between the label and the data for the label to be considered a match. If the ordering of the data doesn't match your expectations, re-write the label to be more specific (labels can be long-winded and descriptive).)")
        st.subheader("Advanced labels")
        st.caption("Labels can be made up of multiple components or imported from the extraction workflow.")
        st.caption("Any label starting with a '!' will be treated a composite label. The field name does not contribute to the actual embedding vector and is just a name for UI purposes. The rest should be a JSON array of strings that will equally contribute to the label.")
        st.text("!my special field['term 1', 'term 2', 'term 3']", help="Each word in the array will be equally weighted in the label. The label name is not used in the vector calculation.")
        st.text("!my special field[-0.006332213724394078, -0.017716300624574813... for 1536 numbers]", help="This is a vector that was exported from the extraction workflow. You can't easily tweak or make these yourself.")

        if apply:
            data_strings = st.session_state.data_strings_raw.split("\n")
            labels_strings = st.session_state.labels_strings_raw.split("\n")
            st.session_state.data_strings = [x.strip() for x in data_strings if x.strip()]
            st.session_state.data_vectors = get_embeddings(st.session_state.data_strings, conn)
            st.session_state.labels_strings = list(map(getFieldName, [x.strip() for x in labels_strings if x.strip()]))
            for label in st.session_state.labels_strings:
                if not label in st.session_state.labels_thresholds:
                    st.session_state.labels_thresholds[label] = 0.18
            st.session_state.labels_vectors = get_embeddings_exp(labels_strings, conn)
            st.session_state.similarity = cosine_similarity(st.session_state.data_vectors, st.session_state.labels_vectors)
            st.session_state.dataframe = None

    canRender = st.session_state.data_vectors is not None and st.session_state.labels_vectors is not None

    with tab3:
        if (not canRender):        
            st.text("Please enter data and labels in the 'Configure' tab and click 'Apply'")
        else:
            col1, col2 = st.columns(2)
            with col1:
                selectedLabel = st.selectbox("Select a label to tune", st.session_state.labels_strings)
            if(selectedLabel):
                with col2:
                    st.session_state.labels_thresholds[selectedLabel] = st.number_input("Threshold", 0.0, 1.0, st.session_state.labels_thresholds[selectedLabel], 0.0001, format="%.4f", key=selectedLabel)
                columns = ['text', 'matches', 'distance']

                data = []
                for i in range(len(st.session_state.data_strings)):
                    thresholdNeeded = 1 - st.session_state.similarity[i][st.session_state.labels_strings.index(selectedLabel)]
                    similarityMatch = thresholdNeeded <= st.session_state.labels_thresholds[selectedLabel]
                    row = [st.session_state.data_strings[i], '✓' if similarityMatch else '', thresholdNeeded]
                    data.append(row)
                dataframe = pd.DataFrame(data, columns=columns)
                dataframe = dataframe.sort_values(by=['distance'], ascending=True)
                st.dataframe(dataframe, hide_index=True, height=1000,
                             column_config={'text' : st.column_config.TextColumn(width="large"),
                                            'matches': st.column_config.TextColumn(width="small"),
                                            'difference' : st.column_config.NumberColumn(format="%.4f", width="small")}, use_container_width=True)
        
    with tab2:
        if (not canRender):        
            st.text("Please enter data and labels in the 'Configure' tab and click 'Apply'")
        else:
            columns = ['text']
            for i in range(len(st.session_state.labels_strings)):
                columns.append(st.session_state.labels_strings[i])

            data = []
            for i in range(len(st.session_state.data_strings)):
                row = [st.session_state.data_strings[i]]
                for j in range(len(st.session_state.labels_strings)):
                    similarityMatch = st.session_state.similarity[i][j] > 1 - st.session_state.labels_thresholds[st.session_state.labels_strings[j]]
                    cellValue = '✓' if similarityMatch else ''
                    row.append(cellValue)
                data.append(row)
            st.session_state.dataframe = pd.DataFrame(data, columns=columns)

            st.dataframe(st.session_state.dataframe, hide_index=True, use_container_width=True)

    conn.close()

if __name__ == "__main__":
    st.set_page_config(layout="wide")
    main()