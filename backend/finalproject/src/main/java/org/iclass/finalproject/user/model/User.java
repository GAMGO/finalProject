package org.iclass.finalproject.user.model;

import jakarta.persistence.*;
import lombok.Getter;
import lombok.Setter;

import java.time.LocalDate;
import java.time.LocalDateTime;

@Entity
@Table(name = "users")
@Getter @Setter
public class User {

    @Id @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(nullable = false, unique = true, length = 50)
    private String username;

    @Column(nullable = false, unique = true, length = 100)
    private String email;

    @Column(name = "password_hash", nullable = false)
    private String passwordHash;

    private LocalDate birthday;
    private String gender;
    private String address;
    private String bio;

    @Column(name = "created_at", nullable = false, updatable = false)
    private LocalDateTime createdAt = LocalDateTime.now();
}

/*
 * [파일 설명]
 * - 서비스 전체에서 사용하는 유저(회원) 엔티티.
 * - 로그인, 리뷰/즐겨찾기/좋아요/장소등록 등 거의 모든 도메인이 이 User를 참조.
 * - DB 테이블 이름은 users, 기본 회원 프로필 정보와 가입 시각을 가진다.
 */
