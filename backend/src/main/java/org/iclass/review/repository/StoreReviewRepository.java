package org.iclass.review.repository;

import org.iclass.review.entity.StoreReview;
import org.springframework.data.domain.*;
import org.springframework.data.jpa.repository.JpaRepository;

public interface StoreReviewRepository extends JpaRepository<StoreReview, Long> {

    Page<StoreReview> findByStore_IdxOrderByCreatedAtDesc(Long storeIdx, Pageable pageable);

    // “한 유저가 같은 가게에 리뷰 1개만” 정책을 쓰고 싶으면 활용
    boolean existsByStore_IdxAndCustomerIdx(Long storeIdx, Long customerIdx);
}
