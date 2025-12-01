package org.iclass.review.entity;

import jakarta.persistence.*;
import java.time.LocalDateTime;

@Entity
@Table(name = "store_review_stats")
public class StoreReviewStats {

    @Id
    @Column(name = "store_idx")
    private Long storeIdx; // PK = STORE.IDX

    @Column(name = "rating_count", nullable = false)
    private Integer ratingCount;

    @Column(name = "avg_rating", nullable = false, precision = 3, scale = 1)
    private Double avgRating;

    // JSON 히스토그램 { "1": n1, "2": n2, ... }
    @Column(name = "rating_histogram", columnDefinition = "JSON")
    private String ratingHistogram;

    @Column(name = "updated_at", nullable = false,
            columnDefinition = "DATETIME")
    private LocalDateTime updatedAt;

    public Long getStoreIdx() { return storeIdx; }
    public void setStoreIdx(Long storeIdx) { this.storeIdx = storeIdx; }
    public Integer getRatingCount() { return ratingCount; }
    public void setRatingCount(Integer ratingCount) { this.ratingCount = ratingCount; }
    public Double getAvgRating() { return avgRating; }
    public void setAvgRating(Double avgRating) { this.avgRating = avgRating; }
    public String getRatingHistogram() { return ratingHistogram; }
    public void setRatingHistogram(String ratingHistogram) { this.ratingHistogram = ratingHistogram; }
    public LocalDateTime getUpdatedAt() { return updatedAt; }
    public void setUpdatedAt(LocalDateTime updatedAt) { this.updatedAt = updatedAt; }
}
