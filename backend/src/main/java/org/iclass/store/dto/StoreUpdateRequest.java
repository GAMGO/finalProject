package org.iclass.store.dto;

import jakarta.validation.constraints.*;
import lombok.*;
import java.time.LocalDateTime;

@Getter
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class StoreUpdateRequest {

    @Size(max = 255, message = "점포 이름은 최대 255자까지 가능합니다.")
    private String storeName;

    private LocalDateTime openTime;

    private LocalDateTime closeTime;

    @Size(max = 255, message = "주소는 최대 255자까지 가능합니다.")
    private String storeAddress;

    @NotNull(message = "음식 카테고리는 필수값입니다.")
    private Long foodTypeId;

    @NotNull(message = "위도(lat)는 필수값입니다.")
    private Double lat;

    @NotNull(message = "경도(lng)는 필수값입니다.")
    private Double lng;
}
