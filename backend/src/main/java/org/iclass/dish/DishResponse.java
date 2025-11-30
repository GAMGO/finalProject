package org.iclass.dish;

import lombok.Getter;
import lombok.AllArgsConstructor;
import lombok.Builder;

/**
 * Dish 응답 DTO
 * - 클라이언트에게 전달할 데이터 전용
 * - setter 없음 (불변 객체)
 */
@Getter
@AllArgsConstructor
@Builder
public class DishResponse {
    private final Long idx;
    private final String name;
    private final String description;
    private final int price;

    /**
     * Entity → DTO 변환 메서드
     */
    public static DishResponse from(Dish dish) {
        return DishResponse.builder()
                .idx(dish.getIdx())
                .name(dish.getName())
                .description(dish.getDescription())
                .price(dish.getPrice())
                .build();
    }
}
