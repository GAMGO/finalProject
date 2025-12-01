package org.iclass.profile.entity;

import jakarta.persistence.*;
import lombok.Getter;
import lombok.Setter;

import java.time.LocalDateTime;

@Getter
@Setter
@Entity
@Table(name = "USER_PROFILE_PAGE")   // 실제 테이블 이름에 맞게 바꿔
public class UserProfileEntity {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long idx;

    // TODO: 나중에 인증 붙으면 CUSTOMERS 외래키
    @Column(name = "customer_idx")
    private Long customerIdx;

    @Column(length = 50)
    private String nickname;

    @Column(length = 255)
    private String intro;

    @Column(name = "avatar_url", length = 1024)
    private String avatarUrl;

    @Column(name = "favorite_food", length = 200)
    private String favoriteFood;

    @Column(length = 200)
    private String location;

    @Column(name = "created_at")
    private LocalDateTime createdAt;

    @Column(name = "updated_at")
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
