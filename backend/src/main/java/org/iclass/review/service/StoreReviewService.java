// src/main/java/org/iclass/review/service/StoreReviewService.java
package org.iclass.review.service;

import jakarta.persistence.EntityNotFoundException;
import lombok.RequiredArgsConstructor;
import org.iclass.review.dto.StoreReviewListWithStatsResponse;
import org.iclass.review.dto.StoreReviewRequest;
import org.iclass.review.dto.StoreReviewResponse;
import org.iclass.review.dto.StoreReviewStatsResponse;
import org.iclass.review.entity.StoreReview;
import org.iclass.review.entity.StoreReviewStats;
import org.iclass.review.repository.StoreReviewRepository;
import org.iclass.review.repository.StoreReviewStatsRepository;
import org.iclass.store.entity.Store;
import org.iclass.store.repository.StoreRepository;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

@Service
@RequiredArgsConstructor
@Transactional(readOnly = true)
public class StoreReviewService {

    private final StoreRepository storeRepository;
    private final StoreReviewRepository reviewRepository;
    private final StoreReviewStatsRepository statsRepository;

    /**
     * 리뷰 생성
     */
    @Transactional
    public Long create(Long storeIdx, Long customerIdx, StoreReviewRequest req) {
        Store store = storeRepository.findById(storeIdx)
                .orElseThrow(() -> new EntityNotFoundException("가게 정보를 찾을 수 없습니다."));

        // 정책: 같은 가게에 동일 사용자의 중복 리뷰 방지
        if (customerIdx != null &&
                reviewRepository.existsByStore_IdxAndCustomerIdx(storeIdx, customerIdx)) {
            throw new IllegalStateException("이미 이 가게에 작성한 리뷰가 있습니다.");
        }

        StoreReview r = new StoreReview();
        r.setStore(store);
        r.setCustomerIdx(customerIdx);
        r.setRating(req.getRating());
        r.setReviewText(req.getReviewText());
        r.setAiTopics(req.getAiTopics());
        // r.setCreatedAt(LocalDateTime.now());    // @PrePersist 에서 처리한다고 가정

        return reviewRepository.save(r).getId();   // 트리거가 통계 갱신
    }

    /**
     * 리뷰 수정
     */
    @Transactional
    public void update(Long reviewId, Long customerIdx, StoreReviewRequest req, boolean isAdmin) {
        StoreReview r = reviewRepository.findById(reviewId)
                .orElseThrow(() -> new EntityNotFoundException("리뷰를 찾을 수 없습니다."));

        // 본인 리뷰만 수정 가능 (관리자는 예외)
        if (!isAdmin && r.getCustomerIdx() != null && !r.getCustomerIdx().equals(customerIdx)) {
            throw new SecurityException("본인 리뷰만 수정할 수 있습니다.");
        }

        r.setRating(req.getRating());
        r.setReviewText(req.getReviewText());
        if (req.getAiTopics() != null) {
            r.setAiTopics(req.getAiTopics());
        }
        // 저장 시 트리거가 통계 자동 반영 (AFTER UPDATE)
    }

    /**
     * 리뷰 삭제
     */
    @Transactional
    public void delete(Long reviewId, Long customerIdx, boolean isAdmin) {
        StoreReview r = reviewRepository.findById(reviewId)
                .orElseThrow(() -> new EntityNotFoundException("리뷰를 찾을 수 없습니다."));

        // 본인 리뷰만 삭제 가능 (관리자는 예외)
        if (!isAdmin && r.getCustomerIdx() != null && !r.getCustomerIdx().equals(customerIdx)) {
            throw new SecurityException("본인 리뷰만 삭제할 수 있습니다.");
        }

        reviewRepository.delete(r);    // AFTER DELETE 트리거가 통계 갱신
    }

    /**
     * 가게별 리뷰 목록 조회 (Page)
     */
    public Page<StoreReviewResponse> list(Long storeIdx, Pageable pageable) {
        Page<StoreReview> p = reviewRepository
                .findByStore_IdxOrderByCreatedAtDesc(storeIdx, pageable);

        return p.map(r -> {
            StoreReviewResponse d = new StoreReviewResponse();
            d.id = r.getId();
            d.storeIdx = r.getStore().getIdx();
            d.customerIdx = r.getCustomerIdx();
            d.rating = r.getRating();
            d.reviewText = r.getReviewText();
            d.blocked = r.getBlocked();
            d.createdAt = (r.getCreatedAt() == null) ? null : r.getCreatedAt().toString();
            d.aiTopics = r.getAiTopics();
            return d;
        });
    }

    /**
     * 가게별 리뷰 통계 조회
     */
    public StoreReviewStatsResponse stats(Long storeIdx) {
        StoreReviewStats s = statsRepository.findById(storeIdx).orElse(null);

        StoreReviewStatsResponse dto = new StoreReviewStatsResponse();
        dto.storeIdx = storeIdx;
        dto.ratingCount = (s == null) ? 0 : s.getRatingCount();
        dto.avgRating = (s == null || s.getAvgRating() == null)
                ? 0.0
                : s.getAvgRating().doubleValue();
        dto.ratingHistogram = (s == null)
                ? "{}"
                : (s.getRatingHistogram() == null ? "{}" : s.getRatingHistogram());

        return dto;
    }

    /**
     * 가게별 리뷰 + 통계 한 번에 조회 (with-stats)
     */
    public StoreReviewListWithStatsResponse listWithStats(Long storeIdx, Pageable pageable) {
        Page<StoreReviewResponse> page = list(storeIdx, pageable);
        StoreReviewStatsResponse statsDto = stats(storeIdx);

        StoreReviewListWithStatsResponse dto = new StoreReviewListWithStatsResponse();
        dto.stats = statsDto;
        dto.reviews = page.getContent();
        return dto;
    }
}
