package org.iclass.recovery.repository;

import org.springframework.data.jpa.repository.JpaRepository;
import org.iclass.recovery.entity.*;

public interface RecoveryRepository extends JpaRepository<Recovery, Long> {
}
