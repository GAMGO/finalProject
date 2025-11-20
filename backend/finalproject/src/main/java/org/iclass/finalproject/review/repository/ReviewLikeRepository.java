package org.iclass.finalproject.review.repository;

import org.iclass.finalproject.review.model.Review;
import org.iclass.finalproject.review.model.ReviewLike;
import org.iclass.finalproject.user.model.User;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.Optional;

public interface ReviewLikeRepository extends JpaRepository<ReviewLike, Long> {
    Optional<ReviewLike> findByReviewAndUser(Review review, User user);
    long countByReview(Review review);
}

/*
 * [파일 설명]
 * - 리뷰 좋아요(ReviewLike) 전용 Repository.
 * - 특정 유저가 이미 좋아요 했는지 확인하고, 현재 좋아요 개수 카운트에 사용.
 */
