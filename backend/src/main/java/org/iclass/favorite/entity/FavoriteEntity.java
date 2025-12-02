package org.iclass.favorite.entity;

import jakarta.persistence.*;
import java.time.LocalDateTime;

@Entity
@Table(name = "favorite")
public class FavoriteEntity {

    public Long getIdx() { return idx; }
    public void setIdx(Long idx) { this.idx = idx; }

    public Long getCustomerIdx() { return customerIdx; }
    public void setCustomerIdx(Long customerIdx) { this.customerIdx = customerIdx; }

    public Long getFavoriteStoreIdx() { return favoriteStoreIdx; }
    public void setFavoriteStoreIdx(Long favoriteStoreIdx) { this.favoriteStoreIdx = favoriteStoreIdx; }

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
