// src/main/java/org/iclass/route/controller/KakaoRouteController.java
package org.iclass.route.controller;

import lombok.Data;
import lombok.RequiredArgsConstructor;
import org.iclass.route.dto.LatLngDto;
import org.iclass.route.dto.RouteSummaryResponse;
import org.iclass.route.service.KakaoRouteService;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api")     // /api/routes
@RequiredArgsConstructor
public class KakaoRouteController {

    private final KakaoRouteService kakaoRouteService;

    /**
     * 프론트에서 호출하는 엔드포인트
     * POST /api/routes
     *
     * {
     *   "from": { "lat": 37.5665, "lng": 126.9780 },
     *   "to":   { "lat": 37.1234, "lng": 127.5678 }
     * }
     */
    @PostMapping("/routes")
    public ResponseEntity<RouteSummaryResponse> getRoute(@RequestBody RouteRequest request) {
        if (request.getFrom() == null || request.getTo() == null) {
            throw new IllegalArgumentException("from/to 좌표가 비어 있습니다.");
        }

        LatLngDto from = request.getFrom();
        LatLngDto to   = request.getTo();

        RouteSummaryResponse result = kakaoRouteService.searchRoute(
                from.getLat(),
                from.getLng(),
                to.getLat(),
                to.getLng()
        );
        return ResponseEntity.ok(result);
    }

    @Data
    public static class RouteRequest {
        private LatLngDto from;
        private LatLngDto to;
    }
}
