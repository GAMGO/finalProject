package org.iclass.dish;

import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;

/**
 * Dish 비즈니스 로직 서비스
 */
@Service
@RequiredArgsConstructor
@Transactional(readOnly = true)
public class DishService {

    private final DishRepository dishRepository;

    /**
     * Dish 생성
     */
    @Transactional
    public DishResponse create(DishRequest req) {
        // 이름 중복 검사
        if (dishRepository.existsByName(req.getName())) {
            throw new IllegalArgumentException("이미 동일한 이름의 메뉴가 존재합니다: " + req.getName());
        }

        // Builder로 엔티티 생성
        Dish dish = Dish.builder()
                .name(req.getName())
                .description(req.getDescription())
                .price(req.getPrice())
                .build();

        Dish saved = dishRepository.save(dish);
        return DishResponse.from(saved);
    }

    /**
     * 모든 Dish 조회
     */
    public List<DishResponse> findAll() {
        return dishRepository.findAll()
                .stream()
                .map(DishResponse::from)
                .toList();
    }

    /**
     * ID로 Dish 조회
     */
    public DishResponse findById(Long idx) {
        Dish dish = dishRepository.findById(idx)
                .orElseThrow(() ->
                        new IllegalArgumentException("해당 ID의 메뉴를 찾을 수 없습니다: " + idx)
                );
        return DishResponse.from(dish);
    }

    /**
     * Dish 수정
     */
    @Transactional
    public DishResponse update(Long idx, DishRequest req) {
        Dish dish = dishRepository.findById(idx)
                .orElseThrow(() ->
                        new IllegalArgumentException("해당 ID의 메뉴를 찾을 수 없습니다: " + idx)
                );

        // 영속 엔티티 Dirty Checking
        dish.setName(req.getName());
        dish.setDescription(req.getDescription());
        dish.setPrice(req.getPrice());

        return DishResponse.from(dish);
    }

    /**
     * Dish 삭제
     */
    @Transactional
    public void delete(Long idx) {
        if (!dishRepository.existsById(idx)) {
            throw new IllegalArgumentException("해당 ID의 메뉴를 찾을 수 없습니다: " + idx);
        }
        dishRepository.deleteById(idx);
    }
}
