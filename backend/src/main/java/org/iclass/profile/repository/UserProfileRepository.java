package org.iclass.profile.repository;

import org.iclass.profile.entity.UserProfileEntity;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.Optional;

public interface UserProfileRepository extends JpaRepository<UserProfileEntity, Long> {

    Optional<UserProfileEntity> findBycustomerIdx(Long customerIdx);
}
