import psycopg2
import numpy as np
import json

class DBClient:
    """A class for interacting with the database. Not thread safe."""
    def __init__(_self, conn):
        _self.conn = conn
        _self.cursor = conn.cursor()

    def __enter__(_self):
        return _self

    def __exit__(_self, exc_type, exc_val, exc_tb):
        _self.cursor.close()

    def get_closest_words(_self, embedding, preferCommonWords=True, k=5):
        embedding_str = ','.join(map(str, embedding))
        embedding_str = f'[{embedding_str}]'
        query = """
        WITH CommonWords AS (
            SELECT word
            FROM words WHERE isCommon = true
            ORDER BY embedding <=> %s
            LIMIT %s
        ),
        UncommonWords AS (
            SELECT word
            FROM words WHERE isCommon = false
            ORDER BY embedding <=> %s
            LIMIT %s
        )

        SELECT word
        FROM (
            SELECT word FROM CommonWords
            UNION ALL
            SELECT word FROM UncommonWords
        ) AS Combined
        LIMIT %s;
        """

        queryAll = """
        SELECT word FROM words
        ORDER BY embedding <=> %s
        LIMIT %s;
        """
        
        if preferCommonWords:
            _self.cursor.execute(query, (embedding_str, k, embedding_str, k, k))
        else:
            _self.cursor.execute(queryAll, (embedding_str, k))
        
        results = _self.cursor.fetchall()
        return [result[0] for result in results]

    def cache_embeddings(_self, hashes, embeddings):
        records_to_insert = [(hash, f'[{",".join(map(str, embedding))}]') for hash, embedding in zip(hashes, embeddings)]
        insert_query = """
        INSERT INTO ada_cache2 (hash, embedding)
        VALUES (%s, %s)
        ON CONFLICT DO NOTHING
        """
        _self.cursor.executemany(insert_query, records_to_insert)
        _self.conn.commit()

    def get_cached_embeddings(_self, hashes):
        query = "SELECT hash, embedding FROM ada_cache2 WHERE hash = ANY(%s)"
        _self.cursor.execute(query, (hashes,))
        return _self.cursor.fetchall()
