package com.facesorter.repository;

import com.facesorter.entity.OrderItemPhoto;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository
public interface OrderItemPhotoRepository extends JpaRepository<OrderItemPhoto, Long> {

    List<OrderItemPhoto> findByOrderItemId(Long orderItemId);
}
