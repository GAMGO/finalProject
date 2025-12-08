package org.iclass.customer.repository;

import org.iclass.customer.entity.CustomersEntity;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;

import java.util.Optional;

public interface CustomersRepository extends JpaRepository<CustomersEntity, Long> {

    // ✅ customer_id(=id 필드)로 엔티티 조회
    Optional<CustomersEntity> findById(String id);
    // ✅ customer_idx로 엔티티 조회
    Optional<CustomersEntity> findByIdx(Long idx);
    
    // ✅ 명시적으로 username(customer_id)으로 찾기
    @Query("SELECT c.idx FROM CustomersEntity c WHERE c.id = :username")
    Optional<Long> findIdxByUsername(@Param("username") String username);

    boolean existsById(String id);
    
    Optional<CustomersEntity> findByEmail(String email);
    // 이메일 인증 토큰으로 사용자 조회
    Optional<CustomersEntity> findByEmailVerificationToken(String token);

    Optional<CustomersEntity> findByRefreshToken(String refreshToken);

    Optional<CustomersEntity> findByIdAndIsDeletedFalse(String id);
}
