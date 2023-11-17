--CREATE TABLE words (word varchar(64) PRIMARY KEY, embedding vector(1536) NOT NULL, isCommon boolean NOT NULL DEFAULT false);
--CREATE TABLE ada_cache (input text PRIMARY KEY, embedding vector(1536) NOT NULL);
--CREATE INDEX idx_vec_words_cs ON words USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);
--CREATE INDEX idx_vec_words_cs ON words USING hnsw (embedding vector_cosine_ops) WITH (m = 32, ef_construction = 120);
--SELECT indexname FROM pg_indexes WHERE tablename = 'words';

CREATE TABLE ada_cache2 (
  id SERIAL PRIMARY KEY,                -- A unique identifier for each row
  hash CHAR(32) NOT NULL UNIQUE,   -- MD5 hash of the text
  embedding vector(1536) NOT NULL       -- The embedding vector of size 1536
);

CREATE INDEX idx_ada_cache2_hash ON ada_cache2 USING btree (hash);

CREATE USER vectoruser WITH PASSWORD '';
GRANT SELECT, INSERT ON ada_cache2 TO vectoruser;
GRANT USAGE, SELECT ON SEQUENCE ada_cache2_id_seq TO vectoruser;
GRANT SELECT ON words TO vectoruser;
