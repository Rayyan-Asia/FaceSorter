package com.facesorter.repository;

import com.facesorter.entity.FaceEmbedding;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository
public interface FaceEmbeddingRepository extends JpaRepository<FaceEmbedding, Long> {

    List<FaceEmbedding> findByEventId(Long eventId);

    /**
     * ANN search using pgvector cosine distance operator.
     * Returns embeddings within the given event ordered by cosine similarity (ascending distance).
     * Lower distance = more similar.
     */
    @Query(value = """
        SELECT fe.* FROM face_embeddings fe
        WHERE fe.event_id = :eventId
          AND (fe.embedding <=> CAST(:queryEmbedding AS vector)) < :threshold
        ORDER BY fe.embedding <=> CAST(:queryEmbedding AS vector)
        LIMIT :maxResults
        """, nativeQuery = true)
    List<FaceEmbedding> findSimilarInEvent(
            @Param("eventId") Long eventId,
            @Param("queryEmbedding") String queryEmbedding,
            @Param("threshold") double threshold,
            @Param("maxResults") int maxResults);

    /**
     * Find all embeddings in an event that are similar to the query, without a result limit.
     */
    @Query(value = """
        SELECT fe.* FROM face_embeddings fe
        WHERE fe.event_id = :eventId
          AND (fe.embedding <=> CAST(:queryEmbedding AS vector)) < :threshold
        ORDER BY fe.embedding <=> CAST(:queryEmbedding AS vector)
        """, nativeQuery = true)
    List<FaceEmbedding> findAllSimilarInEvent(
            @Param("eventId") Long eventId,
            @Param("queryEmbedding") String queryEmbedding,
            @Param("threshold") double threshold);
}
