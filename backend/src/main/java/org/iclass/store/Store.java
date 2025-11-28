package org.iclass.store;

import jakarta.persistence.*;
import java.time.LocalDateTime;
import org.iclass.customer.entity.CustomersEntity;;

@Entity
@Table(name = "STORE")
public class Store {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Column(name = "IDX")
    private Long idx;

    @Column(name = "STORE_NAME", length = 255, nullable = false)
    private String storeName;      // 노점 이름 + 한 줄 설명 느낌

    @Column(name = "OPENTIME")
    private LocalDateTime openTime;   // 아직 안 쓸 거면 null 허용

    @Column(name = "CLOSETIME")
    private LocalDateTime closeTime;

    @Column(name = "STORE_ADDRESS", length = 255, nullable = false)
    private String storeAddress;

    // FK: FOOD_INFO.IDX
    @Column(name = "FOOD_TYPE", nullable = false)
    private Long foodTypeId;

    @Column(name = "LAT", nullable = false)
    private Double lat;

    @Column(name = "LNG", nullable = false)
    private Double lng;

    @Column(name = "INITIAL_RATING")
    private Integer initialRating; // 등록 시 입력하는 별점 (1~5)

    @Column(name = "INITIAL_REVIEW_TITLE", length = 255)
    private String initialReviewTitle; // 등록 시 리뷰 제목

    @Lob
    @Column(name = "INITIAL_REVIEW")
    private String initialReview; // 등록 시 리뷰 내용

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "INITIAL_REVIEW_BY")
    private CustomersEntity initialReviewBy; // 등록자 (고객 FK 참조)

    @Column(name = "INITIAL_REVIEWED_AT")
    private LocalDateTime initialReviewedAt; // 리뷰 작성 시간


    public Store() {}

    // ===== Getter / Setter =====

    public Long getIdx() { return idx; }
    public void setId(Long idx) { this.idx = idx; }

    public String getStoreName() { return storeName; }
    public void setStoreName(String storeName) { this.storeName = storeName; }

    public LocalDateTime getOpenTime() { return openTime; }
    public void setOpenTime(LocalDateTime openTime) { this.openTime = openTime; }

    public LocalDateTime getCloseTime() { return closeTime; }
    public void setCloseTime(LocalDateTime closeTime) { this.closeTime = closeTime; }

    public String getStoreAddress() { return storeAddress; }
    public void setStoreAddress(String storeAddress) { this.storeAddress = storeAddress; }

    public Long getFoodTypeId() { return foodTypeId; }
    public void setFoodTypeId(Long foodTypeId) { this.foodTypeId = foodTypeId; }

    public Double getLat() { return lat; }
    public void setLat(Double lat) { this.lat = lat; }

    public Double getLng() { return lng; }
    public void setLng(Double lng) { this.lng = lng; }

    public Integer getInitialRating() { return initialRating; }
    public void setInitialRating(Integer initialRating) { this.initialRating = initialRating; }

    public String getInitialReviewTitle() { return initialReviewTitle; }
    public void setInitialReviewTitle(String initialReviewTitle) { this.initialReviewTitle = initialReviewTitle; }

    public String getInitialReview() { return initialReview; }
    public void setInitialReview(String initialReview) { this.initialReview = initialReview; }

    public CustomersEntity getInitialReviewBy() { return initialReviewBy; }
    public void setInitialReviewBy(CustomersEntity initialReviewBy) { this.initialReviewBy = initialReviewBy; }

    public LocalDateTime getInitialReviewedAt() { return initialReviewedAt; }
    public void setInitialReviewedAt(LocalDateTime initialReviewedAt) { this.initialReviewedAt = initialReviewedAt; }
}
