-- Replace IVFFlat index with HNSW for cosine similarity search.
-- IVFFlat requires ~lists*39 rows to be trained (lists=100 → ~3900 rows minimum).
-- Querying it with fewer rows causes "No results were returned by the query" errors.
-- HNSW works correctly with any number of rows and has better recall.

DROP INDEX IF EXISTS idx_face_embeddings_vector;

CREATE INDEX idx_face_embeddings_vector ON face_embeddings
    USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);
