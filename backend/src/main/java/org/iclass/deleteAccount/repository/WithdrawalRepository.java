package org.iclass.deleteAccount.repository;

import org.iclass.deleteAccount.entity.WithdrawalEntity;
import org.springframework.data.jpa.repository.JpaRepository;
import java.util.Optional;

public interface WithdrawalRepository extends JpaRepository<WithdrawalEntity, Long> {
    Optional<WithdrawalEntity> findByRecoveryToken(String recoveryToken);
}