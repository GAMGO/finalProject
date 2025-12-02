// src/main/java/org/iclass/route/dto/RouteResponse.java
package org.iclass.route.dto;

import lombok.Data;

import java.util.List;

/**
 * 카카오 내비 원본 응답을 담는 DTO (서비스 내부용)
 */
@Data
public class RouteResponse {
    private double distance;     // m
    private double duration;     // sec
    private Integer taxiFare;    // 원 (null 가능)
    private List<RoutePoint> path;
}
