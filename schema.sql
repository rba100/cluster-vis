SELECT 'read the script' FROM you_pressed_f5_without_reading_the_script;

-- Do not run this whole script. Select individual statements and run
-- them one at a time.

-- install the pgvector extension, minimum version 0.51.0

CREATE TABLE words (word varchar(64) PRIMARY KEY, embedding vector(1536) NOT NULL, isCommon boolean NOT NULL DEFAULT false);

-- Do not build the index until you have populated the words table.
-- Note: This takes a long time, leave it going overnight.
CREATE INDEX idx_vec_words_cs ON words USING hnsw (embedding vector_cosine_ops) WITH (m = 32, ef_construction = 120);

CREATE TABLE ada_cache2 (
  id SERIAL PRIMARY KEY,
  hash CHAR(32) NOT NULL UNIQUE,  -- MD5 hash of the text (not used for security; keep your hair on)
  embedding vector(1536) NOT NULL -- The embedding vector
);

CREATE INDEX idx_ada_cache2_hash ON ada_cache2 USING btree (hash);

CREATE USER vectoruser WITH PASSWORD '';
GRANT SELECT, INSERT ON ada_cache2 TO vectoruser;
GRANT USAGE, SELECT ON SEQUENCE ada_cache2_id_seq TO vectoruser;
GRANT SELECT ON words TO vectoruser;
