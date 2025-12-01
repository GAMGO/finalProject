package org.iclass.review.dto;

public class StoreReviewStatsResponse {
    public Long   storeIdx;
    public Integer ratingCount;
    public Double avgRating;
    public String ratingHistogram; // JSON string (프론트에서 파싱)
}
