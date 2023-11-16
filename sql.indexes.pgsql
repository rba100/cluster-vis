--CREATE TABLE words (word varchar(64) PRIMARY KEY, embedding vector(1536) NOT NULL, isCommon boolean NOT NULL DEFAULT false);
--CREATE TABLE ada_cache (input text PRIMARY KEY, embedding vector(1536) NOT NULL);
--CREATE INDEX idx_vec_words_cs ON words USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);
--CREATE INDEX idx_vec_words_cs ON words USING hnsw (embedding vector_cosine_ops) WITH (m = 32, ef_construction = 120);

SELECT indexname FROM pg_indexes WHERE tablename = 'words';

-- isCommon is a bit
