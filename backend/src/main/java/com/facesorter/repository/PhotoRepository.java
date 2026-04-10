package com.facesorter.repository;

import com.facesorter.entity.Photo;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Set;

@Repository
public interface PhotoRepository extends JpaRepository<Photo, Long> {

    List<Photo> findByEventId(Long eventId);

    List<Photo> findByEventIdAndProcessedFalse(Long eventId);

    long countByEventId(Long eventId);

    long countByEventIdAndProcessedTrue(Long eventId);

    Set<String> findFileHashByEventIdAndFileHashNotNull(Long eventId);
}
