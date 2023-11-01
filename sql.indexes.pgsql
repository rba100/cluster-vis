--CREATE INDEX idx_vec_words_cs ON words USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);

SELECT indexname FROM pg_indexes WHERE tablename = 'words';


SELECT 
    schemaname, 
    tablename, 
    pg_size_pretty(pg_total_relation_size(schemaname || '.' || tablename)) AS total, 
    pg_size_pretty(pg_relation_size(schemaname || '.' || tablename)) AS data, 
    pg_size_pretty(pg_total_relation_size(schemaname || '.' || tablename) - pg_relation_size(schemaname || '.' || tablename)) AS external
FROM pg_tables
ORDER BY pg_total_relation_size(schemaname || '.' || tablename) DESC;

SELECT 
    schemaname,
    indexname,
    pg_size_pretty(pg_relation_size(schemaname || '.' || indexname)) AS size
FROM pg_indexes
ORDER BY pg_relation_size(schemaname || '.' || indexname) DESC;

SELECT 
    oid::regclass::text,
    pg_size_pretty(pg_total_relation_size(oid)) AS total_size,
    pg_size_pretty(pg_relation_size(oid)) AS table_size,
    pg_size_pretty(pg_total_relation_size(oid) - pg_relation_size(oid)) AS toast_size
FROM pg_class
WHERE relkind = 'r'
ORDER BY pg_total_relation_size(oid) DESC;

SELECT 
    schemaname, 
    relname,
    n_live_tup,
    n_dead_tup,
    n_dead_tup * 100.0 / n_live_tup AS dead_percent
FROM pg_stat_user_tables
WHERE n_dead_tup > 0
ORDER BY n_dead_tup DESC;
