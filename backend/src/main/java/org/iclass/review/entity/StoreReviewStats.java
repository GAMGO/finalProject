package org.iclass.review.entity;

import jakarta.persistence.*;
import lombok.Getter;
import lombok.Setter;

import java.math.BigDecimal;
import java.time.LocalDateTime;

@Getter@Setter
@Entity
@Table(name = "store_review_stats")
public class StoreReviewStats {

    @Id
    @Column(name = "store_idx")
    private Long storeIdx; // PK = STORE.IDX

    @Column(name = "rating_count", nullable = false)
    private Integer ratingCount;

    // 0.0 ~ 5.0 범위, 소수점 1자리 예: 4.5
    // ➜ precision/scale 제거해서 Hibernate 6 + MySQLDialect 에러 방지
    @Column(name = "avg_rating", nullable = false)
    private BigDecimal avgRating;

    // JSON 히스토그램 { "1": n1, "2": n2, ... }
    @Column(name = "rating_histogram", columnDefinition = "JSON")
    private String ratingHistogram;

    @Column(name = "updated_at", nullable = false, columnDefinition = "DATETIME")
    private LocalDateTime updatedAt;

}
