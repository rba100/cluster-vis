
def create_extension(conn):
    cursor = conn.cursor()
    cursor.execute("""
    CREATE EXTENSION IF NOT EXISTS vector;
    """)
    conn.commit()
    cursor.close()

def create_table_words(conn):
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE words (
        id SERIAL PRIMARY KEY,
        word VARCHAR(255) UNIQUE NOT NULL,
        isCommon BOOLEAN NOT NULL DEFAULT FALSE
    );
    """)
    conn.commit()
    cursor.close()

def create_table_embeddings(conn, tableName, embeddingSize):
    cursor = conn.cursor()
    cursor.execute(f"""
    CREATE TABLE {tableName} (
        id SERIAL PRIMARY KEY,
        embedding vector({embeddingSize}) NOT NULL,
        FOREIGN KEY (id) REFERENCES words (id)
    );
    """)
    conn.commit()
    cursor.close()

def create_table_embeddings_cache(conn, tableName, embeddingSize):
    cursor = conn.cursor()
    cursor.execute(f"""
    CREATE TABLE {tableName} (
        id SERIAL PRIMARY KEY,
        hash CHAR(32) NOT NULL UNIQUE,  -- MD5 hash of the text (not used for security; keep your hair on)
        embedding vector({embeddingSize}) NOT NULL -- The embedding vector
    );
    """)
    conn.commit()
    cursor.close()

def create_index_embeddings(conn, tableName):
    cursor = conn.cursor()
    cursor.execute(f"""
    CREATE INDEX idx_vec_{tableName}_cs ON {tableName} USING hnsw (embedding vector_ip_ops) WITH (m = 32, ef_construction = 120);
    """)
    conn.commit()
    cursor.close()

def create_table_cache(conn, embeddingTableName, embeddingSize):
    # First create the table
    cursor = conn.cursor()
    cursor.execute(f"""
    CREATE TABLE {embeddingTableName}_cache (
        id SERIAL PRIMARY KEY,
        hash CHAR(32) NOT NULL UNIQUE,
        embedding vector({embeddingSize}) NOT NULL
    );
    """)
    conn.commit()

    # Then create the index
    cursor.execute(f"""
    CREATE INDEX idx_{embeddingTableName}_cache_hash ON {embeddingTableName}_cache USING btree (hash);
    """)
    conn.commit()
    cursor.close()

def create_user(conn, username, password):
    cursor = conn.cursor()
    cursor.execute(f"""
    CREATE USER {username} WITH PASSWORD '{password}';
    """)
    conn.commit()
    cursor.close()

def grant_permissions(conn, username, embeddingTableName):
    cursor = conn.cursor()
    cursor.execute(f"""
    GRANT SELECT, INSERT ON {embeddingTableName}_cache TO {username};
    GRANT USAGE, SELECT ON SEQUENCE {embeddingTableName}_cache_id_seq TO {username};
    GRANT SELECT ON words TO {username};
    """)
    conn.commit()
    cursor.close()