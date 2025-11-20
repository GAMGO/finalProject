package org.iclass.finalproject.review.service;

import lombok.RequiredArgsConstructor;
import org.iclass.finalproject.owner.model.OwnerVerificationStatus;
import org.iclass.finalproject.owner.repository.OwnerVerificationRepository;
import org.iclass.finalproject.place.model.Place;
import org.iclass.finalproject.place.repository.PlaceRepository;
import org.iclass.finalproject.review.dto.ReviewCreateRequest;
import org.iclass.finalproject.review.dto.ReviewResponse;
import org.iclass.finalproject.review.model.Review;
import org.iclass.finalproject.review.model.ReviewLike;
import org.iclass.finalproject.review.repository.ReviewLikeRepository;
import org.iclass.finalproject.review.repository.ReviewRepository;
import org.iclass.finalproject.user.model.User;
import org.iclass.finalproject.user.repository.UserRepository;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDateTime;
import java.util.List;

@Service
@RequiredArgsConstructor
public class ReviewService {

    private final ReviewRepository reviewRepository;
    private final ReviewLikeRepository reviewLikeRepository;
    private final OwnerVerificationRepository ownerVerificationRepository;
    private final UserRepository userRepository;
    private final PlaceRepository placeRepository;

    @Transactional
    public Review addReview(Long userId, Long placeId, ReviewCreateRequest req) {
        User u = userRepository.findById(userId).orElseThrow();
        Place p = placeRepository.findById(placeId).orElseThrow();

        boolean isOwner = ownerVerificationRepository
                .findByPlaceAndUserAndStatus(p, u, OwnerVerificationStatus.APPROVED)
                .isPresent();

        Review r = new Review();
        r.setUser(u);
        r.setPlace(p);
        r.setRating(req.getRating());
        r.setContent(req.getContent());
        r.setOwner(isOwner);
        r.setCreatedAt(LocalDateTime.now());

        r = reviewRepository.save(r);

        // 평점/리뷰 개수 재계산
        List<Review> all = reviewRepository.findByPlaceOrderByCreatedAtDesc(p);
        int count = all.size();
        double avg = all.stream().mapToInt(Review::getRating).average().orElse(0);
        p.setReviewCount(count);
        p.setAvgRating(avg);
        placeRepository.save(p);

        return r;
    }

    @Transactional(readOnly = true)
    public List<ReviewResponse> getReviews(Long placeId) {
        Place p = placeRepository.findById(placeId).orElseThrow();
        return reviewRepository.findByPlaceOrderByCreatedAtDesc(p).stream()
                .map(this::toResponse)
                .toList();
    }

    @Transactional
    public void likeReview(Long userId, Long reviewId) {
        User u = userRepository.findById(userId).orElseThrow();
        Review r = reviewRepository.findById(reviewId).orElseThrow();

        if (reviewLikeRepository.findByReviewAndUser(r, u).isPresent()) return;

        ReviewLike rl = new ReviewLike();
        rl.setReview(r);
        rl.setUser(u);
        reviewLikeRepository.save(rl);

        r.setLikeCount((int) reviewLikeRepository.countByReview(r));
        reviewRepository.save(r);
    }

    @Transactional
    public void unlikeReview(Long userId, Long reviewId) {
        User u = userRepository.findById(userId).orElseThrow();
        Review r = reviewRepository.findById(reviewId).orElseThrow();

        reviewLikeRepository.findByReviewAndUser(r, u)
                .ifPresent(reviewLikeRepository::delete);

        r.setLikeCount((int) reviewLikeRepository.countByReview(r));
        reviewRepository.save(r);
    }

    private ReviewResponse toResponse(Review r) {
        return ReviewResponse.builder()
                .id(r.getId())
                .placeId(r.getPlace().getId())
                .userId(r.getUser().getId())
                .username(r.getUser().getUsername())
                .owner(r.isOwner())
                .rating(r.getRating())
                .content(r.getContent())
                .likeCount(r.getLikeCount())
                .createdAt(r.getCreatedAt())
                .build();
    }
}

/*
 * [파일 설명]
 * - 리뷰 작성/조회/좋아요 처리 및 Place 평점 캐싱(평균/개수)까지 담당하는 서비스.
 * - 사장 인증 여부를 확인해서 owner 플래그를 설정하는 로직도 이곳에 있음.
 */
