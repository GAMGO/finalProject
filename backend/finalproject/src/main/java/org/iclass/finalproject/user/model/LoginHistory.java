// src/main/java/org/iclass/finalproject/user/model/LoginHistory.java
package org.iclass.finalproject.user.model;

import jakarta.persistence.*;
import lombok.Getter;
import lombok.Setter;

import java.time.LocalDate;
import java.time.LocalDateTime;

@Entity
@Table(name = "login_history",
       indexes = {
           @Index(name = "idx_login_history_user_date", columnList = "user_id, loginDate")
       })
@Getter
@Setter
public class LoginHistory {

    @Id @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @ManyToOne(fetch = FetchType.LAZY) @JoinColumn(name = "user_id", nullable = false)
    private User user;

    // 로그인한 정확한 시각
    @Column(nullable = false)
    private LocalDateTime loginAt;

    // 출석 계산을 쉽게 하려고 날짜만 따로 저장
    @Column(nullable = false)
    private LocalDate loginDate;

    public static LoginHistory of(User user) {
        LoginHistory h = new LoginHistory();
        h.setUser(user);
        LocalDateTime now = LocalDateTime.now();
        h.setLoginAt(now);
        h.setLoginDate(now.toLocalDate());
        return h;
    }
}

/*
 * [파일 설명]
 * - 유저가 로그인할 때마다 한 줄씩 기록되는 엔티티.
 * - loginDate 컬럼을 이용해서 총 로그인 일수, 연속 로그인 일수 계산.
 */
