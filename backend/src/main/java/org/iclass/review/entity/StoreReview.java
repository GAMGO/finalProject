package org.iclass.review.entity;

import jakarta.persistence.*;
import java.time.LocalDateTime;
import org.iclass.store.entity.Store;

@Entity
@Table(name = "store_reviews",
       indexes = {
         @Index(name = "ix_sr_store_created", columnList = "store_idx, created_at DESC"),
         @Index(name = "ix_sr_store_rating",  columnList = "store_idx, rating")
       })
public class StoreReview {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Column(name = "idx")
    private Long id;

    @ManyToOne(fetch = FetchType.LAZY, optional = false)
    @JoinColumn(name = "store_idx", nullable = false,
            foreignKey = @ForeignKey(name = "fk_sr_store"))
    private Store store;

    @Column(name = "customer_idx")
    private Long customerIdx; // FK -> CUSTOMERS.IDX (NULL 허용)

    @Column(name = "rating", nullable = false)
    private Integer rating;   // TINYINT(1~5)

    @Lob
    @Column(name = "review_text")
    private String reviewText;

    @Column(name = "is_blocked", nullable = false)
    private Boolean blocked = false;

    @Column(name = "created_at", nullable = false,
            columnDefinition = "DATETIME")
    private LocalDateTime createdAt; // DB DEFAULT CURRENT_TIMESTAMP 사용

    @Column(name = "ai_topics", columnDefinition = "JSON")
    private String aiTopics;

    @Column(name = "sentiment_score", columnDefinition = "FLOAT")
    private Float sentimentScore; // 감정 점수 (NULL 허용)

    @Column(name = "sentiment_label", length = 20)
    private String sentimentLabel; // 감정 라벨 (NULL 허용)

    /* === getters/setters === */
    public Long getId() { return id; }

    public Store getStore() { return store; }
    public void setStore(Store store) { this.store = store; }

    public Long getCustomerIdx() { return customerIdx; }
    public void setCustomerIdx(Long customerIdx) { this.customerIdx = customerIdx; }

    public Integer getRating() { return rating; }
    public void setRating(Integer rating) { this.rating = rating; }

    public String getReviewText() { return reviewText; }
    public void setReviewText(String reviewText) { this.reviewText = reviewText; }

    public Boolean getBlocked() { return blocked; }
    public void setBlocked(Boolean blocked) { this.blocked = blocked; }

    public LocalDateTime getCreatedAt() { return createdAt; }
    public void setCreatedAt(LocalDateTime createdAt) { this.createdAt = createdAt; }

    public String getAiTopics() { return aiTopics; }
    public void setAiTopics(String aiTopics) { this.aiTopics = aiTopics; }

    public Float getSentimentScore() { return sentimentScore; }
    public void setSentimentScore(Float sentimentScore) { this.sentimentScore = sentimentScore; }

    public String getSentimentLabel() { return sentimentLabel; }
    public void setSentimentLabel(String sentimentLabel) { this.sentimentLabel = sentimentLabel; }

    @PrePersist
    protected void onCreate() {
        if (this.createdAt == null) {              // 생성 시점 자동 세팅
            this.createdAt = LocalDateTime.now();
        }
        if (this.blocked == null) {                // null 방지(안전장치)
            this.blocked = false;
        }
    }
}
