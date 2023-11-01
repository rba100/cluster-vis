import psycopg2
import openai
import numpy as np

def get_embeddings(text_list, conn):
    batch_size = 500
    all_embeddings = []
    
    # Create a cursor object to interact with the database
    cursor = conn.cursor()

    for i in range(0, len(text_list), batch_size):
        batch = text_list[i:i + batch_size]

        # Check if embeddings are already cached in the database
        placeholders = ', '.join(['%s'] * len(batch))

        query = f"SELECT input, embedding FROM ada_cache WHERE input IN ({placeholders})"
        cursor.execute(query, tuple(batch))
        cached_results = cursor.fetchall()

        # Separate the cached and uncached items
        cached_dict = {item[0]: item[1] for item in cached_results}
        uncached_texts = [text for text in batch if text not in cached_dict]

        # Fetch embeddings for uncached items
        if uncached_texts:
            result = openai.Embedding.create(model='text-embedding-ada-002', input=uncached_texts)
            uncached_embeddings = [item['embedding'] for item in result['data']]

            # Store the new embeddings in the database for future use
            for text, embedding in zip(uncached_texts, uncached_embeddings):
                embedding_str = ','.join(map(str, embedding))
                embedding_str = f'[{embedding_str}]'
                cursor.execute("INSERT INTO ada_cache (input, embedding) VALUES (%s, %s)", (text, embedding_str))
            conn.commit()

            all_embeddings.extend(uncached_embeddings)

        # Add the cached embeddings
        all_embeddings.extend([list(map(float, cached_dict[text][1:-1].split(','))) for text in batch if text in cached_dict])

    # Close the cursor
    cursor.close()

    return all_embeddings
