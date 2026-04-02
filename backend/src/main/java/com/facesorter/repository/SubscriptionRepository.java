package com.facesorter.repository;

import com.facesorter.entity.Subscription;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Optional;

@Repository
public interface SubscriptionRepository extends JpaRepository<Subscription, Long> {

    List<Subscription> findByStudioId(Long studioId);

    Optional<Subscription> findByStudioIdAndActiveTrue(Long studioId);

    boolean existsByStudioIdAndActiveTrue(Long studioId);

    long countByActiveTrue();
}
