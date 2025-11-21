// src/main/java/org/iclass/finalproject/user/model/User.java
package org.iclass.finalproject.user.model;

import jakarta.persistence.*;
import lombok.Getter;
import lombok.Setter;

import java.time.LocalDate;
import java.time.LocalDateTime;

@Entity
@Table(name = "users",
       uniqueConstraints = {
           @UniqueConstraint(columnNames = "email")
       })
@Getter
@Setter
public class User {

    @Id @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    // 사용자 이름(닉네임 느낌)
    @Column(nullable = false, length = 50)
    private String username;

    // 로그인용 이메일
    @Column(nullable = false, length = 100)
    private String email;

    // 암호화된 비밀번호
    @Column(name = "password_hash", nullable = false, length = 255)
    private String passwordHash;

    // 선택 정보들
    private LocalDate birthday;

    @Column(length = 10)
    private String gender;

    @Column(length = 255)
    private String address;

    @Column(length = 500)
    private String bio;

    // 기본 타임스탬프
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;

    @PrePersist
    protected void onCreate() {
        this.createdAt = LocalDateTime.now();
        this.updatedAt = this.createdAt;
    }

    @PreUpdate
    protected void onUpdate() {
        this.updatedAt = LocalDateTime.now();
    }
}

/*
 * [파일 설명]
 * - 회원 정보를 저장하는 User 엔티티.
 * - 로그인(이메일/비번), 프로필(이름/생일/성별/주소/소개) 등
 *   네가 요구한 설정 페이지의 "회원정보" 파트가 이 테이블에서 나온다.
 */
