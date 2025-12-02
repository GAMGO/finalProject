// src/main/java/org/iclass/route/controller/KakaoRouteController.java
package org.iclass.route.controller;

import lombok.Data;
import lombok.RequiredArgsConstructor;
import org.iclass.route.dto.RouteSummaryResponse;
import org.iclass.route.service.KakaoRouteService;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api")     // 🔹 전체 prefix: /api
@RequiredArgsConstructor
public class KakaoRouteController {

    private final KakaoRouteService kakaoRouteService;

    /**
     * 프론트에서 호출하는 엔드포인트
     * POST /api/routes
     *
     * {
     *   "originLat": 37.5665,
     *   "originLng": 126.9780,
     *   "destLat": 37.1234,
     *   "destLng": 127.5678
     * }
     */
    @PostMapping("/routes")
    public ResponseEntity<RouteSummaryResponse> getRoute(@RequestBody RouteRequest request) {
        RouteSummaryResponse result = kakaoRouteService.searchRoute(
                request.getOriginLat(),
                request.getOriginLng(),
                request.getDestLat(),
                request.getDestLng()
        );
        return ResponseEntity.ok(result);
    }

    @Data
    public static class RouteRequest {
        private double originLat;
        private double originLng;
        private double destLat;
        private double destLng;
    }
}
