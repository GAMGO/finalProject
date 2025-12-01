package org.iclass.dish;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

@Repository
public interface DishRepository extends JpaRepository<Dish, Long> {

    /**
     * 이름 중복 여부 확인
     * @param name Dish 이름
     * @return true: 이미 존재, false: 존재하지 않음
     */
    boolean existsByName(String name);
}
