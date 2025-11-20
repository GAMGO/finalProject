package org.iclass.finalproject.review.repository;

import org.iclass.finalproject.place.model.Place;
import org.iclass.finalproject.review.model.Review;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.List;
import java.util.Optional;

public interface ReviewRepository extends JpaRepository<Review, Long> {

    List<Review> findByPlaceOrderByCreatedAtDesc(Place place);

    Optional<Review> findTopByPlaceOrderByLikeCountDesc(Place place);
}

/*
 * [파일 설명]
 * - 리뷰 목록/대표 리뷰 조회를 담당하는 Repository.
 * - 장소별 최신순 목록과 "좋아요 가장 많은 리뷰" 조회 메서드를 제공.
 */
