// src/main/java/org/iclass/finalproject/user/repository/UserRepository.java
package org.iclass.finalproject.user.repository;

import org.iclass.finalproject.user.model.User;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.Optional;

public interface UserRepository extends JpaRepository<User, Long> {

    Optional<User> findByEmail(String email);
}

/*
 * [파일 설명]
 * - User 엔티티에 대한 기본 CRUD + 이메일로 유저 찾기 기능 제공.
 * - 로그인, 비밀번호 재설정 등에서 사용.
 */
