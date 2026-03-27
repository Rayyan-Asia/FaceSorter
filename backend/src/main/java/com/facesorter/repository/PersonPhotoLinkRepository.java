package com.facesorter.repository;

import com.facesorter.entity.PersonPhotoLink;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository
public interface PersonPhotoLinkRepository extends JpaRepository<PersonPhotoLink, Long> {

    List<PersonPhotoLink> findByFaceEmbeddingId(Long faceEmbeddingId);

    List<PersonPhotoLink> findByPhotoId(Long photoId);

    boolean existsByFaceEmbeddingIdAndPhotoId(Long faceEmbeddingId, Long photoId);

    @Query("SELECT ppl FROM PersonPhotoLink ppl WHERE ppl.faceEmbedding.id IN :embeddingIds")
    List<PersonPhotoLink> findByFaceEmbeddingIdIn(@Param("embeddingIds") List<Long> embeddingIds);
}
