package org.iclass.finalproject.place.repository;

import org.iclass.finalproject.place.model.Place;
import org.iclass.finalproject.place.model.PlaceCategory;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.List;

public interface PlaceRepository extends JpaRepository<Place, Long> {

    List<Place> findByCategory(PlaceCategory category);
}

/*
 * [파일 설명]
 * - Place 엔티티에 대한 기본 DB 접근 인터페이스.
 * - 카테고리별 검색 등 간단한 조회 메서드를 제공.
 * - 위치 기반 검색은 나중에 커스텀 쿼리/Native 쿼리로 확장할 예정.
 */
