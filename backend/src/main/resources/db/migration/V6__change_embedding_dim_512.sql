-- Switch from FaceNet 128-dim to InsightFace ArcFace 512-dim embeddings.
-- ArcFace (via InsightFace + ONNX Runtime DirectML) provides better accuracy
-- and GPU acceleration on Windows without a CUDA toolkit.
-- All existing embeddings are cleared — they must be regenerated with the new model.

TRUNCATE person_photo_links, face_embeddings, photos, orders, order_items, order_item_photos RESTART IDENTITY CASCADE;

DROP INDEX IF EXISTS idx_face_embeddings_vector;

ALTER TABLE face_embeddings DROP COLUMN embedding;
ALTER TABLE face_embeddings ADD COLUMN embedding vector(512) NOT NULL;

CREATE INDEX idx_face_embeddings_vector ON face_embeddings
    USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);
