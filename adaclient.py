import psycopg2
import openai
import numpy as np

def get_embeddings(text_list, conn):
    batch_size = 500
    all_embeddings = []
    embeddings_dict = {}
    
    # Create a cursor object to interact with the database
    cursor = conn.cursor()

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
                result = openai.Embedding.create(model='text-embedding-ada-002', input=uncached_texts)
            except Exception as e:
                print(uncached_texts)
                raise e
                        
            uncached_embeddings = [item['embedding'] for item in result['data']]

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
    conn.commit()

    return np.array(all_embeddings)
