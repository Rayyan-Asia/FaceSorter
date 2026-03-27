package com.facesorter.repository;

import com.facesorter.entity.Studio;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.Optional;

@Repository
public interface StudioRepository extends JpaRepository<Studio, Long> {

    Optional<Studio> findByEmail(String email);

    Optional<Studio> findByGoogleSub(String googleSub);

    boolean existsByEmail(String email);
}
