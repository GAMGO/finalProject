// CustomersRepository.java

package org.iclass.customer.repository;

import org.iclass.customer.entity.CustomersEntity;
import org.iclass.customer.entity.Gender;

import java.util.Optional;

import org.springframework.data.jpa.repository.JpaRepository;

public interface CustomersRepository extends JpaRepository<CustomersEntity, Long> {
  
  Optional<CustomersEntity> findById(String customer_id);
  Optional<CustomersEntity> findByEmail(String email);

  boolean existsById(String customer_id);

  // 이메일 인증 토큰으로 사용자 조회
  Optional<CustomersEntity> findByEmailVerificationToken(String token);

  // 🚨이모지로 표시: 매개변수 타입을 String에서 Gender로 변경합니다.
  Optional<CustomersEntity> findByIdxAndGender(Long id, Gender gender);
  Optional<CustomersEntity> findByIdx(Long id);
}