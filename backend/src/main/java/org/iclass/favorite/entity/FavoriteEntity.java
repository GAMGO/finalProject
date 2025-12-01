package org.iclass.favorite.entity;

import jakarta.persistence.*;
import lombok.Getter;
import lombok.Setter;

import java.time.LocalDateTime;
@Getter
@Setter
@Entity
@Table(name = "FAVORITE")   // 실제 테이블명 다르면 여기만 바꾸면 됨
public class FavoriteEntity {


    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long idx;

    // TODO: 로그인 붙으면 CUSTOMER_ID 외래키로 교체
    @Column(name = "customer_idx")
    private Long customerIdx;

    @Column(name = "FAVORITE_STORE_IDX")
    private Long favoriteStoreIdx;

    @Column(nullable = false, length = 50)
    private String category;

    @Column(nullable = false, length = 200)
    private String title;

    @Column(length = 255)
    private String address;

    @Column(columnDefinition = "TEXT")
    private String note;

    @Column
    private Double rating;

    @Column(name = "image_url", length = 1024)
    private String imageUrl;

    @Column(name = "video_url", length = 1024)
    private String videoUrl;

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
