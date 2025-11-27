package org.iclass.favorite.repository;

import org.iclass.favorite.entity.FavoriteEntity;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.List;

public interface FavoriteRepository extends JpaRepository<FavoriteEntity, Long> {

    List<FavoriteEntity> findByCustomerIdOrderByCreatedAtDesc(Long idx);
}
