package org.iclass.finalproject.dish;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

@Repository
public interface DishRepository extends JpaRepository<dish, Long> {
    boolean existsByName(String name);
}
