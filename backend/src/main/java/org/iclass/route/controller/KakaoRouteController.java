// src/main/java/org/iclass/route/controller/KakaoRouteController.java
package org.iclass.route.controller;

import lombok.RequiredArgsConstructor;
import org.iclass.common.ApiResponse;
import org.iclass.route.dto.RouteResponse;
import org.iclass.route.service.KakaoRouteService;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/routes")
@RequiredArgsConstructor
public class KakaoRouteController {

    private final KakaoRouteService kakaoRouteService;

    @GetMapping
    public ApiResponse<RouteResponse> getRoute(
            @RequestParam double originLat,
            @RequestParam double originLng,
            @RequestParam double destLat,
            @RequestParam double destLng
    ) {
        RouteResponse route = kakaoRouteService.getRoute(
                originLat, originLng, destLat, destLng
        );
        return ApiResponse.success(route);
    }
}
