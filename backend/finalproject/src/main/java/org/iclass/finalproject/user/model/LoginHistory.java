package org.iclass.finalproject.user.model;

import jakarta.persistence.*;
import lombok.Getter;
import lombok.Setter;

import java.time.LocalDateTime;

@Entity
@Table(name = "login_history")
@Getter @Setter
public class LoginHistory {

    @Id @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "user_id", nullable = false)
    private User user;

    @Column(name = "login_at", nullable = false)
    private LocalDateTime loginAt;

    public static LoginHistory of(User user) {
        LoginHistory h = new LoginHistory();
        h.setUser(user);
        h.setLoginAt(LocalDateTime.now());
        return h;
    }
}

/*
 * [파일 설명]
 * - 회원별 로그인 시각을 기록하는 엔티티.
 * - 매 로그인 성공 시 한 줄씩 insert해서
 *   "총 로그인 일수 / 연속 로그인 / 출석 달력" 같은 기능에 사용.
 */
