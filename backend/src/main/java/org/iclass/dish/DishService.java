package org.iclass.dish;

import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;

@Service
@RequiredArgsConstructor            // final 필드 기반 생성자 자동 생성
@Transactional(readOnly = true)
public class DishService {

    private final DishRepository dishRepository;

    /**
     * Dish 생성
     */
    @Transactional
    public DishResponse create(DishRequest req) {
        // 이름 중복 체크 (DB에도 unique 제약 권장: @Column(unique = true))
        if (dishRepository.existsByName(req.getName())) {
            throw new IllegalArgumentException("Dish with the same name already exists: " + req.getName());
        }

        // ✅ Builder 사용해 엔티티 생성 (idx는 자동 생성)
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
     * ID로 단건 조회
     */
    public DishResponse findById(Long idx) {
        Dish dish = dishRepository.findById(idx)
                .orElseThrow(() -> new IllegalArgumentException("Dish not found: " + idx));
        return DishResponse.from(dish);
    }

    /**
     * Dish 수정
     */
    @Transactional
    public DishResponse update(Long idx, DishRequest req) {
        Dish dish = dishRepository.findById(idx)
                .orElseThrow(() -> new IllegalArgumentException("Dish not found: " + idx));

        // 영속 엔티티에 값만 세팅하면 Dirty Checking으로 update 반영됨
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
            throw new IllegalArgumentException("Dish not found: " + idx);
        }
        dishRepository.deleteById(idx);
    }
}
