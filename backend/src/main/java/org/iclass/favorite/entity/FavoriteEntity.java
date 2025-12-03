package org.iclass.favorite.entity;

import jakarta.persistence.*;
import java.time.LocalDateTime;

@Entity
@Table(name = "favorite")   // DDL 이 FAVORITE 여도 MySQL 기본 설정에선 대소문자 무시
public class FavoriteEntity {

    /* ========================= Getter / Setter ========================= */

    public Long getIdx() { return idx; }
    public void setIdx(Long idx) { this.idx = idx; }

    public Long getCustomerIdx() { return customerIdx; }
    public void setCustomerIdx(Long customerIdx) { this.customerIdx = customerIdx; }

    public Long getFavoriteStoreIdx() { return favoriteStoreIdx; }
    public void setFavoriteStoreIdx(Long favoriteStoreIdx) { this.favoriteStoreIdx = favoriteStoreIdx; }

    public String getCategory() { return category; }
    public void setCategory(String category) { this.category = category; }

    public String getTitle() { return title; }
    public void setTitle(String title) { this.title = title; }

    public String getAddress() { return address; }
    public void setAddress(String address) { this.address = address; }

    public String getNote() { return note; }
    public void setNote(String note) { this.note = note; }

    public Double getRating() { return rating; }
    public void setRating(Double rating) { this.rating = rating; }

    public String getImageUrl() { return imageUrl; }
    public void setImageUrl(String imageUrl) { this.imageUrl = imageUrl; }

    public String getVideoUrl() { return videoUrl; }
    public void setVideoUrl(String videoUrl) { this.videoUrl = videoUrl; }

    public LocalDateTime getCreatedAt() { return createdAt; }
    public void setCreatedAt(LocalDateTime createdAt) { this.createdAt = createdAt; }

    public LocalDateTime getUpdatedAt() { return updatedAt; }
    public void setUpdatedAt(LocalDateTime updatedAt) { this.updatedAt = updatedAt; }

    /* ========================= Fields ========================= */

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long idx;

    @Column(name = "customer_idx")
    private Long customerIdx;

    @Column(name = "FAVORITE_STORE_IDX")
    private Long favoriteStoreIdx;

    // ☆ 카테고리
    @Column(name = "CATEGORY", length = 50)
    private String category;

    // ☆ 상호 / 이름
    @Column(name = "TITLE", length = 200)
    private String title;

    // ☆ 위치
    @Column(name = "ADDRESS", length = 255)
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
