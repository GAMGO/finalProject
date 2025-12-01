package org.iclass.dish;

import jakarta.validation.constraints.*;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.AllArgsConstructor;
import lombok.Builder;

/**
 * Dish 생성/수정 요청 DTO
 */
@Getter
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class DishRequest {

    @NotBlank(message = "메뉴 이름은 비워둘 수 없습니다.")
    @Size(max = 100, message = "메뉴 이름은 최대 100자까지 가능합니다.")
    private String name;

    @Size(max = 1000, message = "설명은 최대 1000자까지 가능합니다.")
    private String description;

    @Positive(message = "가격은 0보다 큰 양수여야 합니다.")
    private int price;
}
