package org.iclass.userlocation.repository;

import org.springframework.data.jpa.repository.JpaRepository;
import org.iclass.userlocation.entity.*;

public interface UserLocationRepository extends JpaRepository<UserLocation, Long> {
}
