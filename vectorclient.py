import openai
import numpy as np
import streamlit as st
import json

@st.cache_data(max_entries=500)
def get_embeddings(text_list, _conn):
    batch_size = 500
    all_embeddings = []
    embeddings_dict = {}
    
    # Create a cursor object to interact with the database
    cursor = _conn.cursor()

    # De-dupe the text_list
    text_list_deduped = list(set(text_list))

    for i in range(0, len(text_list_deduped), batch_size):
        batch = text_list_deduped[i:i + batch_size]

        query = "SELECT input, embedding FROM ada_cache WHERE input = ANY(%s)"
        cursor.execute(query, (batch,))
        cached_results = cursor.fetchall()

        # Separate the cached and uncached items
        cached_dict = {item[0]: item[1] for item in cached_results}
        uncached_texts = [text for text in batch if text not in cached_dict]

        # Fetch embeddings for uncached items
        if uncached_texts:
            try:
                result = openai.embeddings.create(model='text-embedding-ada-002', input=uncached_texts)
            except Exception as e:
                raise e
                        
            uncached_embeddings = [item.embedding for item in result.data]

            # Store the new embeddings in the database for future use
            for text, embedding in zip(uncached_texts, uncached_embeddings):
                embedding_str = ','.join(map(str, embedding))
                embedding_str = f'[{embedding_str}]'
                #cursor.execute("INSERT INTO ada_cache (input, embedding) VALUES (%s, %s)", (text, embedding_str))
                cursor.execute("INSERT INTO ada_cache (input, embedding) VALUES (%s, %s) ON CONFLICT DO NOTHING", (text, embedding_str))                

            embeddings_dict.update({text: embedding for text, embedding in zip(uncached_texts, uncached_embeddings)})

        # Add the cached embeddings to the dictionary
        embeddings_dict.update({text: list(map(float, cached_dict[text][1:-1].split(','))) for text in batch if text in cached_dict})

    # Map the original text_list into the dictionary to get embeddings in order
    all_embeddings = [embeddings_dict[text] for text in text_list]

    # Close the cursor
    cursor.close()
    _conn.commit()

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
    # else if the list items are numbers that can be cast to floats
    elif all(isinstance(item, int) or isinstance(item, float) for item in terms):
        if(len(terms) != 1536):
            raise Exception("Expression vector must have 1536 dimensions")
        return [np.array(terms)]
    else:
        raise Exception("Expression string must include an array of strings or numbers")
    mean = np.mean(vectors, axis=0)
    return mean / np.linalg.norm(mean)

def getFieldName(input: str):
    if(len(input) < 2 or input[0] != '!'):
        return input    
    openBraceIndex = input.find('[')
    if openBraceIndex == -1:
        return input
    return input[1:openBraceIndex].strip()
