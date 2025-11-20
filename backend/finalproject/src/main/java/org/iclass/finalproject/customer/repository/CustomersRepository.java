// CustomersRepository.java

package org.iclass.finalproject.customer.repository;

import org.iclass.finalproject.customer.entity.CustomersEntity;
import org.iclass.finalproject.customer.entity.Gender;

import java.util.Optional;

import org.springframework.data.jpa.repository.JpaRepository;

public interface CustomersRepository extends JpaRepository<CustomersEntity, Long> {
  
  Optional<CustomersEntity> findById(String id);

  boolean existsById(String id);

  // 이메일 인증 토큰으로 사용자 조회
  Optional<CustomersEntity> findByEmailVerificationToken(String token);

  // 🚨이모지로 표시: 매개변수 타입을 String에서 Gender로 변경합니다.
  Optional<CustomersEntity> findByIdxAndGender(Long idx, Gender gender);
  Optional<CustomersEntity> findByIdx(Long idx);
}