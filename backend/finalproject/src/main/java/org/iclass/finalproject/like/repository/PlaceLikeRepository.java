package org.iclass.finalproject.like.repository;

import org.iclass.finalproject.like.model.PlaceLike;
import org.iclass.finalproject.place.model.Place;
import org.iclass.finalproject.user.model.User;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.Optional;

public interface PlaceLikeRepository extends JpaRepository<PlaceLike, Long> {
    Optional<PlaceLike> findByUserAndPlace(User user, Place place);
    long countByPlace(Place place);
}

/*
 * [파일 설명]
 * - 가게 좋아요(PlaceLike) 전용 Repository.
 * - 유저별 좋아요 여부 확인 및 해당 가게의 좋아요 개수 조회 기능 제공.
 */
