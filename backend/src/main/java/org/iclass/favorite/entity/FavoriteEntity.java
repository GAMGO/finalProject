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

    // ===== getter / setter =====

    // public Long getIdx() {
    //     return idx;
    // }

    // public void setId(Long idx) {
    //     this.idx = idx;
    // }

    // public Long getcustomerIdx() {
    //     return customerIdx;
    // }

    // public void setcustomerIdx(Long customerIdx) {
    //     this.customerIdx = customerIdx;
    // }

    // public String getCategory() {
    //     return category;
    // }

    // public void setCategory(String category) {
    //     this.category = category;
    // }

    // public String getTitle() {
    //     return title;
    // }

    // public void setTitle(String title) {
    //     this.title = title;
    // }

    // public String getAddress() {
    //     return address;
    // }

    // public void setAddress(String address) {
    //     this.address = address;
    // }

    // public String getNote() {
    //     return note;
    // }

    // public void setNote(String note) {
    //     this.note = note;
    // }

    // public Double getRating() {
    //     return rating;
    // }

    // public void setRating(Double rating) {
    //     this.rating = rating;
    // }

    // public String getImageUrl() {
    //     return imageUrl;
    // }

    // public void setImageUrl(String imageUrl) {
    //     this.imageUrl = imageUrl;
    // }

    // public String getVideoUrl() {
    //     return videoUrl;
    // }

    // public void setVideoUrl(String videoUrl) {
    //     this.videoUrl = videoUrl;
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
