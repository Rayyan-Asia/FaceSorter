package com.facesorter.repository;

import com.facesorter.entity.User;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.Optional;

@Repository
public interface UserRepository extends JpaRepository<User, Long> {

    Optional<User> findByIdNumber(String idNumber);

    Optional<User> findByGoogleSub(String googleSub);

    boolean existsByIdNumber(String idNumber);
}
