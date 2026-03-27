package com.facesorter.repository;

import com.facesorter.entity.Order;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository
public interface OrderRepository extends JpaRepository<Order, Long> {

    List<Order> findByEventId(Long eventId);

    List<Order> findByStudioId(Long studioId);

    List<Order> findByUserId(Long userId);

    List<Order> findByStudioIdAndStatus(Long studioId, String status);
}
