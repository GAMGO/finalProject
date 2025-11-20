package org.iclass.finalproject.review.controller;

import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.iclass.finalproject.common.ApiResponse;
import org.iclass.finalproject.common.CurrentUser;
import org.iclass.finalproject.review.dto.ReviewCreateRequest;
import org.iclass.finalproject.review.dto.ReviewResponse;
import org.iclass.finalproject.review.service.ReviewService;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/api/places/{placeId}/reviews")
@RequiredArgsConstructor
public class ReviewController {

    private final ReviewService reviewService;

    @PostMapping
    public ApiResponse<Long> addReview(@PathVariable Long placeId,
                                       @RequestBody @Valid ReviewCreateRequest req) {
        Long userId = CurrentUser.getUserId();
        var review = reviewService.addReview(userId, placeId, req);
        return ApiResponse.ok(review.getId());
    }

    @GetMapping
    public ApiResponse<List<ReviewResponse>> listReviews(@PathVariable Long placeId) {
        return ApiResponse.ok(reviewService.getReviews(placeId));
    }

    @PostMapping("/{reviewId}/like")
    public ApiResponse<Void> likeReview(@PathVariable Long placeId,
                                        @PathVariable Long reviewId) {
        Long userId = CurrentUser.getUserId();
        reviewService.likeReview(userId, reviewId);
        return ApiResponse.ok(null);
    }

    @DeleteMapping("/{reviewId}/like")
    public ApiResponse<Void> unlikeReview(@PathVariable Long placeId,
                                          @PathVariable Long reviewId) {
        Long userId = CurrentUser.getUserId();
        reviewService.unlikeReview(userId, reviewId);
        return ApiResponse.ok(null);
    }
}

/*
 * [파일 설명]
 * - 리뷰 CRUD 중 "작성/조회 + 좋아요"에 해당하는 REST 엔드포인트.
 * - 장소 상세 페이지에서 리뷰 탭/댓글 기능과 직접 연결되는 컨트롤러.
 */
