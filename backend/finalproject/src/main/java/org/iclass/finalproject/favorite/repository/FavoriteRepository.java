package org.iclass.finalproject.favorite.repository;

import org.iclass.finalproject.favorite.model.Favorite;
import org.iclass.finalproject.place.model.Place;
import org.iclass.finalproject.user.model.User;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.List;
import java.util.Optional;

public interface FavoriteRepository extends JpaRepository<Favorite, Long> {
    Optional<Favorite> findByUserAndPlace(User user, Place place);
    List<Favorite> findByUser(User user);
}

/*
 * [파일 설명]
 * - Favorite 엔티티 전용 Repository.
 * - 특정 유저의 즐겨찾기 목록 조회 및 단일 즐겨찾기 존재 여부 확인에 사용.
 */
