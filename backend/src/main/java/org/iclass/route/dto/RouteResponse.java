// src/main/java/org/iclass/route/dto/RouteResponse.java
package org.iclass.route.dto;

import lombok.Data;

import java.util.List;

@Data
public class RouteResponse {
    private double distance;     // m
    private double duration;     // sec
    private Integer taxiFare;    // 원 (null 가능)
    private List<RoutePoint> path;
}
