package org.iclass.favorite.repository;

import org.iclass.favorite.entity.FavoriteEntity;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.List;

public interface FavoriteRepository extends JpaRepository<FavoriteEntity, Long> {

    // ☆ 메서드명 수정: ByCustomerIdx...
    List<FavoriteEntity> findByCustomerIdxOrderByCreatedAtDesc(Long customerIdx);
}
