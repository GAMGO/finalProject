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
 * - 특정 유저의 보안 질문 목록 조회, 질문 종류별 단건 조회 등을 제공하는 레포.
 * - 비밀번호 재설정 시 보안 질문/답 검증에 사용.
 */
