import streamlit as st
from vectorclient import get_embeddings, get_embeddings_exp, getFieldName, reflect_vector
from sklearn.metrics.pairwise import cosine_similarity
from st_utils import value_persister
import numpy as np
import pandas as pd
import psycopg2

def classify_load_data(conn):
    data_strings = st.session_state.data_strings_raw.split("\n")
    labels_strings = st.session_state.labels_strings_raw.split("\n")
    st.session_state.data_strings = [x.strip() for x in data_strings if x.strip()]
    st.session_state.data_vectors = get_embeddings(st.session_state.data_strings, conn)
    if(st.session_state.removeConceptText.strip() != ""):
        vectorToRemove = get_embeddings([st.session_state.removeConceptText.strip()], conn)[0]
        st.session_state.data_vectors = np.apply_along_axis(reflect_vector, 1, st.session_state.data_vectors, vectorToRemove)
    st.session_state.labels_strings = list(map(getFieldName, [x.strip() for x in labels_strings if x.strip()]))
    for label in st.session_state.labels_strings:
        if not label in st.session_state.labels_thresholds:
            st.session_state.labels_thresholds[label] = 0.18
    st.session_state.labels_vectors = get_embeddings_exp(labels_strings, conn)
    st.session_state.similarity = cosine_similarity(st.session_state.data_vectors, st.session_state.labels_vectors)
    st.session_state.dataframe = None

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
        st.session_state.removeConceptText = st.text_input("Remove concept from data", value=st.session_state.removeConceptText, help="If you have a concept that is common to all the items you can enter it here and then clustering will try to ignore that sentiment. For example, if you have a list of comments about 'car problems' you don't want the clustering to be dominated by the word 'car'. This is a feature that can really mess up the clustering if you enter text which isn't common to all text items because they will be modified as if they were which could take them literally anywhere in multidimentional vector space. When experimenting with this, try turninig off OpenAI cluster naming so you can see the underlying cluster concepts.")

        labelStringsKey, labelStringsUpdate = value_persister("labels_strings_raw")
        st.text_area("Enter your labels", key=labelStringsKey, on_change=labelStringsUpdate)
        submitText = "Submit" if st.session_state.data_vectors is None else "Apply"
        apply = st.button(submitText)

        st.subheader("Instructions")
        st.caption("Your labels will be sematically matched to the data. You can see how a label matches to the data by adjusting the threshold in the 'Tune' tab. The threshold is the minimum similarity between the label and the data for the label to be considered a match. If the ordering of the data doesn't match your expectations, re-write the label to be more specific (labels can be long-winded and descriptive).)")
        st.subheader("Advanced labels")
        st.caption("Labels can be made up of multiple components or imported from the extraction workflow.")
        st.caption("Any label starting with a '!' will be treated a composite label. The field name does not contribute to the actual embedding vector and is just a name for UI purposes. The rest should be a JSON array of strings that will equally contribute to the label.")
        st.text("!my special field['term 1', 'term 2', 'term 3']", help="Each word in the array will be equally weighted in the label. The label name is not used in the vector calculation.")
        st.text("!my special field[-0.006332213724394078, -0.017716300624574813... for 1536 numbers]", help="This is a vector that was exported from the extraction workflow. You can't easily tweak or make these yourself.")

        if apply:
            classify_load_data(conn)

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