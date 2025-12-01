// src/main/java/org/iclass/route/dto/RouteSummaryResponse.java
package org.iclass.route.dto;

import lombok.Data;

import java.util.List;

/**
 * 프론트에 리턴해줄 길찾기 요약 + 경로 정보 DTO
 */
@Data
public class RouteSummaryResponse {

    /** 전체 거리 (미터) */
    private Integer distance;

    /** 예상 소요 시간 (초) */
    private Integer duration;

    /** 예상 택시 요금 (원) - 없으면 null */
    private Integer taxiFare;

    /** 유료도로 요금 (원) - 없으면 null */
    private Integer tollFare;

    /** 경로 좌표 리스트 (lat, lng) */
    private List<LatLngDto> path;
}
