import openai
import numpy as np
import streamlit as st
import json
import hashlib

debug = False

def md5_hash(text):
    # Return MD5 hash of the given text
    return hashlib.md5(text.encode('utf-8')).hexdigest()

@st.cache_data(max_entries=4)
def get_embeddings(lines, _conn):
    if debug:
        print("get_embeddings: " + str(len(lines)))

    cursor = _conn.cursor()
    text_hash_mapping = {md5_hash(text): text for text in lines}
    hashed_lines = list(text_hash_mapping.keys())

    # No need to dedupe since we're using a set to create the hash mapping
    batch_size = 500
    all_embeddings = []
    embeddings_dict = {}

    for i in range(0, len(hashed_lines), batch_size):
        batch_hashes = hashed_lines[i:i + batch_size]

        query = "SELECT hash, embedding FROM ada_cache2 WHERE hash = ANY(%s)"
        if debug:
            print("checking cache")
        cursor.execute(query, (batch_hashes,))
        cached_results = cursor.fetchall()

        # Separate the cached and uncached items
        cached_hashes = {item[0]: item[1] for item in cached_results}
        uncached_hashes = [hash for hash in batch_hashes if hash not in cached_hashes]

        if uncached_hashes:
            uncached_texts = [text_hash_mapping[hash] for hash in uncached_hashes]
            if debug:
                print("uncached: " + str(len(uncached_texts)))
            try:
                if debug:
                    print("calling openai")
                result = openai.embeddings.create(model='text-embedding-ada-002', input=uncached_texts)
            except Exception as e:
                raise e
                        
            uncached_embeddings = [item.embedding for item in result.data]

            if debug:
                print("inserting into cache")

            records_to_insert = [
                (hash, f'[{",".join(map(str, embedding))}]')
                for hash, embedding in zip(uncached_hashes, uncached_embeddings)
            ]

            insert_query = """
            INSERT INTO ada_cache2 (hash, embedding)
            VALUES (%s, %s)
            ON CONFLICT DO NOTHING
            """

            cursor.executemany(insert_query, records_to_insert)
            _conn.commit()
            embeddings_dict.update({hash: embedding for hash, embedding in zip(uncached_hashes, uncached_embeddings)})

        embeddings_dict.update({hash: list(map(float, cached_hashes[hash][1:-1].split(','))) for hash in batch_hashes if hash in cached_hashes})

    all_embeddings = [embeddings_dict[md5_hash(text)] for text in lines]
    cursor.close()
    _conn.commit()

    if debug:
        print("returning embeddings")

    if(len(all_embeddings) != len(lines)):
        raise Exception("Number of embeddings does not match number of lines")

    return np.array(all_embeddings)

def reflect_vector(normal, target):
    # Reflects target across the plane orthogonal to normal
    # Both vectors should be normalized and have the same dimension
    return -target + 2 * np.dot(target, normal) * normal

def getMidVector(v1, v2):
    # Calculate the vector that is the sum of v1 and v2
    sum_vector = v1 + v2
    # Normalize the sum_vector to find the halfway vector
    normalized_halfway_vector = sum_vector / np.linalg.norm(sum_vector)
    return normalized_halfway_vector

st.cache_data(max_entries=5)
def get_embeddings_exp(items, _conn):
    non_composites = [item for item in items if not item.startswith('!')]
    non_composite_vectors = get_embeddings(non_composites, _conn) if non_composites else []
    non_composite_iter = iter(non_composite_vectors)
    vectors = [getCompositeVector(item, _conn) if item.startswith('!') else next(non_composite_iter) for item in items]
    
    return vectors

# expressionString must be a string
st.cache_data(max_entries=5)

def getCompositeVector(expressionString: str, _conn):
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
        vectors = get_embeddings(terms, _conn)
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
