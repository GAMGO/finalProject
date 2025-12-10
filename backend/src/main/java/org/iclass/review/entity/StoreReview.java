package org.iclass.review.entity;

import jakarta.persistence.*;
import lombok.Getter;
import lombok.Setter;

import java.time.LocalDateTime;
import org.iclass.store.entity.Store;

@Getter
@Setter
@Entity
@Table(name = "store_reviews", indexes = {
        @Index(name = "ix_sr_store_created", columnList = "store_idx, created_at DESC"),
        @Index(name = "ix_sr_store_rating", columnList = "store_idx, rating")
})
public class StoreReview {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Column(name = "idx")
    private Long id;

    @ManyToOne(fetch = FetchType.LAZY, optional = false)
    @JoinColumn(name = "store_idx", nullable = false, foreignKey = @ForeignKey(name = "fk_sr_store"))
    private Store store;

    @Column(name = "customer_idx")
    private Long customerIdx; // FK -> CUSTOMERS.IDX (NULL 허용)

    @Column(name = "rating", nullable = false)
    private Integer rating; // TINYINT(1~5)

    @Lob
    @Column(name = "review_text")
    private String reviewText;

    @Column(name = "is_blocked", nullable = false, columnDefinition = "TINYINT(1) DEFAULT 0")
    private Boolean blocked = false;

    @Column(name = "created_at", nullable = false, updatable = false, columnDefinition = "DATETIME DEFAULT CURRENT_TIMESTAMP")
    private LocalDateTime createdAt; // DB DEFAULT CURRENT_TIMESTAMP 사용

    @Column(name = "ai_topics", columnDefinition = "JSON")
    private String aiTopics;

    @Column(name = "sentiment_score", columnDefinition = "FLOAT")
    private Float sentimentScore; // 감정 점수 (NULL 허용)

    @Column(name = "sentiment_label", length = 20)
    private String sentimentLabel; // 감정 라벨 (NULL 허용)

    @PrePersist
    protected void onCreate() {
        if (this.createdAt == null) { // 생성 시점 자동 세팅
            this.createdAt = LocalDateTime.now();
        }
        if (this.blocked == null) { // null 방지(안전장치)
            this.blocked = false;
        }
    }
}
