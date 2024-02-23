from openai import OpenAI 
import numpy as np
import streamlit as st
import json
import hashlib
from dbclient import DBClient

client = OpenAI()

def md5_hash(text):
    # Return MD5 hash of the given text
    return hashlib.md5(text.encode('utf-8')).hexdigest()

@st.cache_data(max_entries=4, experimental_allow_widgets=True, show_spinner=False)
def get_embeddings_st(lines, _dbClient: DBClient, showProgress=False):
    return get_embeddings(lines, _dbClient, showProgress)

def get_embeddings(lines, _dbClient: DBClient, showProgress=False):

    text_hash_mapping = {md5_hash(text): text for text in lines}
    hashed_lines = list(text_hash_mapping.keys())

    batch_size = 500
    all_embeddings = []
    embeddings_dict = {}

    if showProgress:
        embeddingsProgress = st.progress(0, text="Encoding text items...")

    for i in range(0, len(hashed_lines), batch_size):
        if showProgress and i != 0:
            embeddingsProgress.progress(i / len(hashed_lines), text="Encoding text items...")
        batch_hashes = hashed_lines[i:i + batch_size]

        cached_results =_dbClient.get_cached_embeddings(batch_hashes)

        # Separate the cached and uncached items
        cached_hashes = {item[0]: item[1] for item in cached_results}
        uncached_hashes = [hash for hash in batch_hashes if hash not in cached_hashes]

        if uncached_hashes:
            uncached_texts = [text_hash_mapping[hash] for hash in uncached_hashes]

            try:
                result = client.embeddings.create(model='text-embedding-ada-002', input=uncached_texts)
            except Exception as e:
                raise e
                        
            uncached_embeddings = [item.embedding for item in result.data]

            records_to_insert = [
                (hash, f'[{",".join(map(str, embedding))}]')
                for hash, embedding in zip(uncached_hashes, uncached_embeddings)
            ]

            _dbClient.cache_embeddings(uncached_hashes, uncached_embeddings)
            embeddings_dict.update({hash: embedding for hash, embedding in zip(uncached_hashes, uncached_embeddings)})

        embeddings_dict.update({hash: list(map(float, cached_hashes[hash][1:-1].split(','))) for hash in batch_hashes if hash in cached_hashes})
    if showProgress:
        embeddingsProgress.empty()
    all_embeddings = [embeddings_dict[md5_hash(text)] for text in lines]

    if(len(all_embeddings) != len(lines)):
        raise Exception("Number of embeddings does not match number of lines")

    return np.array(all_embeddings)

def reflect_vector(normal, target):
    # Reflects target across the plane orthogonal to normal
    # Both vectors should be normalized and have the same dimension
    return -target + 2 * np.dot(target, normal) * normal

def getMidVector(v1, v2):
    sum_vector = v1 + v2
    normalized_halfway_vector = sum_vector / np.linalg.norm(sum_vector)
    return normalized_halfway_vector

st.cache_data(max_entries=5)
def get_embeddings_exp(items, _dbClient: DBClient, _progressCallback=None):
    non_composites = [item for item in items if not item.startswith('!')]
    non_composite_vectors = get_embeddings_st(non_composites, _dbClient, _progressCallback) if non_composites else []
    non_composite_iter = iter(non_composite_vectors)
    vectors = [getCompositeVector(item, _dbClient) if item.startswith('!') else next(non_composite_iter) for item in items]
    
    return vectors

# expressionString must be a string
st.cache_data(max_entries=5)

def getCompositeVector(expressionString: str, _dbClient: DBClient):
    if(len(expressionString) < 1):
        raise Exception("Expression string must be at least one character long")
    if(expressionString[0] != '!'):
        raise Exception("Expression string must start with !")
    
    openBraceIndex = expressionString.find('[')
    fieldName = expressionString[1:openBraceIndex].strip()
    terms = json.loads(expressionString[openBraceIndex:])
    # assert is an array
    if not isinstance(terms, list):
        raise Exception("Expression string must end with a JSON array")

    # if list items are strings
    if all(isinstance(item, str) for item in terms):
        vectors = get_embeddings_st(terms, _dbClient)
        mean = np.mean(vectors, axis=0)
        return mean / np.linalg.norm(mean)
    # else if the list items are numbers that can be cast to floats
    elif all(isinstance(item, int) or isinstance(item, float) for item in terms):
        if(len(terms) != 1536):
            raise Exception("Expression vector must have 1536 dimensions")
        return list(np.array(terms))
    else:
        raise Exception("Expression string must include an array of strings or numbers")

def getFieldName(input: str):
    if(len(input) < 2 or input[0] != '!'):
        return input    
    openBraceIndex = input.find('[')
    if openBraceIndex == -1:
        return input
    return input[1:openBraceIndex].strip()
