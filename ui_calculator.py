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

def connectToDb():
    return psycopg2.connect(st.secrets["connectionString"])

def main():
    input = st.text_input("Enter a single text example.")
    conceptToRemove = st.text_input("Enter a concept to remove from the text.")

    canCompute = input.strip() != ""

    if canCompute:
        conn = connectToDb()
        inputVector = get_embeddings([input], conn)[0]
        finalVector = inputVector
        if conceptToRemove.strip() != "":
            conceptToRemoveVector = get_embeddings([conceptToRemove], conn)[0]
            finalVector = reflect_vector(inputVector, conceptToRemoveVector)
        cursor = conn.cursor()
        nearestWords = get_closest_words(finalVector,cursor, k=20)
        st.text("\n".join(nearestWords))
        cursor.close()
        conn.close()

if __name__ == "__main__":
    main()