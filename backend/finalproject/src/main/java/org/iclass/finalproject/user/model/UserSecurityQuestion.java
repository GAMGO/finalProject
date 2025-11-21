// src/main/java/org/iclass/finalproject/user/model/UserSecurityQuestion.java
package org.iclass.finalproject.user.model;

import jakarta.persistence.*;
import lombok.Getter;
import lombok.Setter;

@Entity
@Table(name = "user_security_questions",
       uniqueConstraints = @UniqueConstraint(columnNames = {"user_id", "question"}))
@Getter
@Setter
public class UserSecurityQuestion {

    @Id @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @ManyToOne(fetch = FetchType.LAZY) @JoinColumn(name = "user_id", nullable = false)
    private User user;

    @Enumerated(EnumType.STRING)
    @Column(nullable = false, length = 30)
    private SecurityQuestionType question;

    @Column(name = "answer_hash", nullable = false, length = 255)
    private String answerHash;
}

/*
 * [파일 설명]
 * - 각 유저가 설정한 보안 질문/답변을 저장하는 테이블.
 * - 답변은 평문이 아니라 해시로 저장해서 보안 유지.
 */
