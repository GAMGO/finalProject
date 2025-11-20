package org.iclass.finalproject.user.repository;

import org.iclass.finalproject.user.model.User;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.Optional;

public interface UserRepository extends JpaRepository<User, Long> {
    Optional<User> findByEmail(String email);
}

/*
 * [파일 설명]
 * - User 엔티티에 대한 JPA Repository.
 * - CRUD + 이메일로 회원 찾기 기능 제공.
 * - 서비스 레이어에서 회원 조회/저장을 담당하는 DAO 역할.
 */
