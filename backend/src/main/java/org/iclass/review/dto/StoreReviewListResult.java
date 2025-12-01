package org.iclass.review.dto;

import java.util.List;

public class StoreReviewListResult {

    // 리뷰 통계
    public StoreReviewStatsResponse stats;

    // 리뷰 목록
    public List<StoreReviewResponse> reviews;

    public StoreReviewListResult() {
    }

    public StoreReviewListResult(StoreReviewStatsResponse stats,
                                 List<StoreReviewResponse> reviews) {
        this.stats = stats;
        this.reviews = reviews;
    }
}
