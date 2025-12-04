// org.iclass.recovery.repository.RecoveryRepository
package org.iclass.recovery.repository;

import java.util.Optional;

import org.iclass.customer.entity.CustomersEntity;
import org.iclass.recovery.entity.Recovery;
import org.springframework.data.jpa.repository.JpaRepository;

public interface RecoveryRepository extends JpaRepository<Recovery, Long> {

    Optional<Recovery> findByCustomer(CustomersEntity customer);
}

// 회원가입과 회원정보를 묶기 위한 보강