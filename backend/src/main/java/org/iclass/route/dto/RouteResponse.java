// src/main/java/org/iclass/route/dto/RouteResponse.java
package org.iclass.route.dto;

import lombok.Data;

import java.util.List;

@Data
public class RouteResponse {
    private Double distance;          // m
    private Double duration;          // sec (우리가 변환해서 넣음)
    private Integer taxiFare;         // 택시 요금
    private Integer tollFare;         // 톨게이트 요금 (없으면 null)
    private List<RoutePoint> path;    // 경로 좌표들 (lat,lng)
}
