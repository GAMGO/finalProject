// src/main/java/org/iclass/finalproject/user/repository/UserSecurityQuestionRepository.java
package org.iclass.finalproject.user.repository;

import org.iclass.finalproject.user.model.SecurityQuestionType;
import org.iclass.finalproject.user.model.User;
import org.iclass.finalproject.user.model.UserSecurityQuestion;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.List;
import java.util.Optional;

public interface UserSecurityQuestionRepository extends JpaRepository<UserSecurityQuestion, Long> {

    List<UserSecurityQuestion> findByUser(User user);

    Optional<UserSecurityQuestion> findByUserAndQuestion(User user, SecurityQuestionType question);
}

/*
 * [파일 설명]
 * - UserSecurityQuestion 엔티티에 대한 DB 접근 레이어.
 * - 특정 유저의 전체 질문 목록 및 질문 종류별 단건 조회 기능 제공.
 */
