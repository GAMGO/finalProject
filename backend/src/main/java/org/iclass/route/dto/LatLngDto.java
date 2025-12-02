// src/main/java/org/iclass/route/dto/LatLngDto.java
package org.iclass.route.dto;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * 위도/경도 좌표 DTO (프론트로 내려줄 때 사용)
 */
@Data
@NoArgsConstructor
@AllArgsConstructor
public class LatLngDto {
    private double lat;   // 위도
    private double lng;   // 경도
}
