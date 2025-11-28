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

    // // --- getter / setter ---

    // public Long getIdx() {
    //     return idx;
    // }

    // public void setIdx(Long idx) {
    //     this.idx = idx;
    // }

    // public Long getcustomerIdx() {
    //     return customerIdx;
    // }

    // public void setcustomerIdx(Long customerIdx) {
    //     this.customerIdx = customerIdx;
    // }

    // public String getNickname() {
    //     return nickname;
    // }

    // public void setNickname(String nickname) {
    //     this.nickname = nickname;
    // }

    // public String getIntro() {
    //     return intro;
    // }

    // public void setIntro(String intro) {
    //     this.intro = intro;
    // }

    // public String getAvatarUrl() {
    //     return avatarUrl;
    // }

    // public void setAvatarUrl(String avatarUrl) {
    //     this.avatarUrl = avatarUrl;
    // }

    // public String getFavoriteFood() {
    //     return favoriteFood;
    // }

    // public void setFavoriteFood(String favoriteFood) {
    //     this.favoriteFood = favoriteFood;
    // }

    // public String getLocation() {
    //     return location;
    // }

    // public void setLocation(String location) {
    //     this.location = location;
    // }

    // public LocalDateTime getCreatedAt() {
    //     return createdAt;
    // }

    // public void setCreatedAt(LocalDateTime createdAt) {
    //     this.createdAt = createdAt;
    // }

    // public LocalDateTime getUpdatedAt() {
    //     return updatedAt;
    // }

    // public void setUpdatedAt(LocalDateTime updatedAt) {
    //     this.updatedAt = updatedAt;
    // }
}
