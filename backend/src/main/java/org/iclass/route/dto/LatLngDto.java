// src/main/java/org/iclass/route/dto/LatLngDto.java
package org.iclass.route.dto;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * 위도/경도 좌표 DTO
 */
@Data
@NoArgsConstructor
@AllArgsConstructor
public class LatLngDto {
    private double lat;   // 위도
    private double lng;   // 경도
}
