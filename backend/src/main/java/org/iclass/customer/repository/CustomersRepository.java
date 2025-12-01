package org.iclass.customer.repository;

import org.iclass.customer.entity.CustomersEntity;
import org.iclass.customer.entity.Gender;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;

import java.util.Optional;

public interface CustomersRepository extends JpaRepository<CustomersEntity, Long> {

    // ✅ customer_id(=id 필드)로 엔티티 조회
    Optional<CustomersEntity> findById(String id); // 그대로 둬도 되지만 이름 충돌 우려됨

    // ✅ 명시적으로 username(customer_id)으로 찾기
    @Query("SELECT c.idx FROM CustomersEntity c WHERE c.id = :username")
    Optional<Long> findIdxByUsername(@Param("username") String username);

    Optional<CustomersEntity> findByEmail(String email);

    boolean existsById(String id);

    // 이메일 인증 토큰으로 사용자 조회
    Optional<CustomersEntity> findByEmailVerificationToken(String token);

    // 🚨이모지로 표시: 매개변수 타입을 String에서 Gender로 변경합니다.
    Optional<CustomersEntity> findByIdxAndGender(Long idx, Gender gender);

    Optional<CustomersEntity> findByIdx(Long idx);
}
