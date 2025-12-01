package org.iclass.review.repository;

import org.iclass.review.entity.StoreReviewStats;
import org.springframework.data.jpa.repository.JpaRepository;

public interface StoreReviewStatsRepository extends JpaRepository<StoreReviewStats, Long> {
    // PK = store_idx
}
