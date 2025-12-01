package org.iclass.review.service;

import jakarta.persistence.EntityNotFoundException;
import org.iclass.review.dto.*;
import org.iclass.review.entity.StoreReview;
import org.iclass.review.entity.StoreReviewStats;
import org.iclass.review.repository.*;
import org.iclass.review.repository.*;
import org.iclass.store.Store;
import org.iclass.store.StoreRepository;
import org.springframework.data.domain.*;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDateTime;

@Service
public class StoreReviewService {

    private final StoreRepository storeRepository;
    private final StoreReviewRepository reviewRepository;
    private final StoreReviewStatsRepository statsRepository;

    public StoreReviewService(StoreRepository storeRepository,
                              StoreReviewRepository reviewRepository,
                              StoreReviewStatsRepository statsRepository) {
        this.storeRepository = storeRepository;
        this.reviewRepository = reviewRepository;
        this.statsRepository = statsRepository;
    }

    @Transactional
    public Long create(Long storeIdx, Long customerIdx, StoreReviewRequest req) {
        Store store = storeRepository.findById(storeIdx)
                .orElseThrow(() -> new EntityNotFoundException("Store not found"));

        // 정책: 같은 가게에 동일 사용자의 중복 리뷰 방지(원치 않으면 제거)
        if (customerIdx != null &&
            reviewRepository.existsByStore_IdxAndCustomerIdx(storeIdx, customerIdx)) {
            throw new IllegalStateException("이미 이 가게에 작성한 리뷰가 있습니다.");
        }

        StoreReview r = new StoreReview();
        r.setStore(store);
        r.setCustomerIdx(customerIdx);
        r.setRating(req.rating);
        r.setReviewText(req.reviewText);
        r.setCreatedAt(LocalDateTime.now()); // DB DEFAULT 있지만 명시 저장도 OK
        r.setAiTopics(req.aiTopics);

        return reviewRepository.save(r).getId(); // 트리거가 통계 갱신
    }

    @Transactional
    public void update(Long reviewId, Long customerIdx, StoreReviewRequest req, boolean isAdmin) {
        StoreReview r = reviewRepository.findById(reviewId)
                .orElseThrow(() -> new EntityNotFoundException("Review not found"));

        if (!isAdmin && r.getCustomerIdx() != null && !r.getCustomerIdx().equals(customerIdx)) {
            throw new SecurityException("본인 리뷰만 수정할 수 있습니다.");
        }
        r.setRating(req.rating);
        r.setReviewText(req.reviewText);
        if (req.aiTopics != null) r.setAiTopics(req.aiTopics);
        // 저장 시 트리거가 통계 자동 반영 (AFTER UPDATE)
    }

    @Transactional
    public void delete(Long reviewId, Long customerIdx, boolean isAdmin) {
        StoreReview r = reviewRepository.findById(reviewId)
                .orElseThrow(() -> new EntityNotFoundException("Review not found"));

        if (!isAdmin && r.getCustomerIdx() != null && !r.getCustomerIdx().equals(customerIdx)) {
            throw new SecurityException("본인 리뷰만 삭제할 수 있습니다.");
        }
        reviewRepository.delete(r); // AFTER DELETE 트리거가 통계 갱신
    }

    @Transactional(readOnly = true)
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
            d.createdAt = r.getCreatedAt() == null ? null : r.getCreatedAt().toString();
            d.aiTopics = r.getAiTopics();
            return d;
        });
    }

    @Transactional(readOnly = true)
    public StoreReviewStatsResponse stats(Long storeIdx) {
        StoreReviewStats s = statsRepository.findById(storeIdx).orElse(null);
        StoreReviewStatsResponse dto = new StoreReviewStatsResponse();
        dto.storeIdx = storeIdx;
        dto.ratingCount = s == null ? 0 : s.getRatingCount();
        dto.avgRating = s == null ? 0.0 : s.getAvgRating();
        dto.ratingHistogram = s == null ? "{}" : (s.getRatingHistogram() == null ? "{}" : s.getRatingHistogram());
        return dto;
    }
}
