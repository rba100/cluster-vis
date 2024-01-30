import streamlit as st
from vectorclient import get_embeddings, get_embeddings_exp, getFieldName, reflect_vector
from sklearn.metrics.pairwise import cosine_similarity
from st_utils import value_persister, init_session_state
from dbclient import DBClient
import numpy as np
import pandas as pd
import psycopg2

def classify_load_data(_dbClient: DBClient):
    data_strings = st.session_state.data_strings_raw.split("\n")
    labels_strings = st.session_state.labels_strings_raw.split("\n")
    st.session_state.data_strings = [x.strip() for x in data_strings if x.strip()]
    st.session_state.data_vectors = get_embeddings(st.session_state.data_strings, _dbClient)
    st.session_state.labels_strings = list(map(getFieldName, [x.strip() for x in labels_strings if x.strip()]))
    for label in st.session_state.labels_strings:
        if not label in st.session_state.labels_thresholds:
            st.session_state.labels_thresholds[label] = 0.18
    st.session_state.labels_vectors = get_embeddings_exp(labels_strings, _dbClient)
    st.session_state.similarity = cosine_similarity(st.session_state.data_vectors, st.session_state.labels_vectors)
    st.session_state.dataframe = None

def main():

    init_session_state(empty   = ['data_strings', 'labels_strings', 'data_vectors', 'labels_vectors', 'dataframe', 'similarity'],
                       strings = ['data_strings_raw', 'labels_strings_raw', 'removeConceptText'],
                       dicts   = ['labels_thresholds'])

    def connectToDb():
        return psycopg2.connect(st.secrets["connectionString"])

    conn = connectToDb()
    with DBClient(conn) as _dbClient:

        tab1, tab2, tab3 = st.tabs(["Configure", "Tabulate", "Tune"])

        with tab1:
            st.session_state.data_strings_raw = st.text_area("Enter your text items", value=st.session_state.data_strings_raw)
            labelStringsKey, labelStringsUpdate = value_persister("labels_strings_raw")
            st.text_area("Enter your labels", key=labelStringsKey, on_change=labelStringsUpdate)
            submitText = "Submit" if st.session_state.data_vectors is None else "Apply"
            apply = st.button(submitText)

            st.subheader("Instructions")
            st.caption("Your labels will be sematically matched to the data. You can see how a label matches to the data by adjusting the threshold in the 'Tune' tab. The threshold is the minimum similarity between the label and the data for the label to be considered a match. If the ordering of the data doesn't match your expectations, re-write the label to be more specific (labels can be long-winded and descriptive).")
            st.subheader("Advanced labels")
            st.caption("Labels can be made up of multiple components or imported from the extraction workflow.")
            st.caption("Any label starting with a '!' will be treated a composite label. The field name does not contribute to the actual embedding vector and is just a name for UI purposes. The rest should be a JSON array of strings that will equally contribute to the label.")
            st.text("!my special field['term 1', 'term 2', 'term 3']", help="Each word in the array will be equally weighted in the label. The label name is not used in the vector calculation.")
            st.text("!my special field[-0.006332213724394078, -0.017716300624574813... for 1536 numbers]", help="This is a vector that was exported from the extraction workflow. You can't easily tweak or make these yourself.")

            if apply:
                classify_load_data(_dbClient)

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
                        key, update = value_persister(selectedLabel, storeKey='labels_thresholds')
                        st.number_input("Threshold", 0.0, 1.0, format="%.4f", key=key, on_change=update)
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