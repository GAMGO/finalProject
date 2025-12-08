package org.iclass.route.service;

import lombok.RequiredArgsConstructor;
import org.iclass.route.dto.LatLngDto;
import org.iclass.route.dto.RouteResponse;
import org.iclass.route.dto.RouteSummaryResponse;
import org.iclass.route.dto.TransportMode;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.stream.Collectors;

@Service
@RequiredArgsConstructor
public class KakaoRouteService {

    // 🔥 이제 실제 호출은 네이버로 보냄
    private final NaverDirectionsService naverDirectionsService;

    /**
     * 프론트에서 쓰는 메인 메서드
     */
    public RouteSummaryResponse searchRoute(
            TransportMode mode,
            double originLat,
            double originLng,
            double destLat,
            double destLng
    ) {
        if (mode == null) mode = TransportMode.CAR;

        LatLngDto from = new LatLngDto(originLat, originLng);
        LatLngDto to   = new LatLngDto(destLat, destLng);

        // 네이버 Directions 호출
        RouteResponse raw = naverDirectionsService.getRoute(from, to, mode);

        int distance = (int) Math.round(raw.getDistance());   // m
        int baseDuration = (int) Math.round(raw.getDuration()); // sec (NaverDirectionsService에서 이미 /1000 해줬음)

        int durationSec;
        switch (mode) {
            case WALK:
                // 도보는 거리 기반으로 다시 계산
                double walkSpeed = 1.3; // m/s
                durationSec = (int) Math.round(distance / walkSpeed);
                break;
            case TRANSIT:
                // 대중교통은 일단 자동차 시간 + α 로 보정 (필요하면 나중에 수정)
                durationSec = baseDuration + 5 * 60;
                break;
            case CAR:
            default:
                durationSec = baseDuration;
        }

        RouteSummaryResponse summary = new RouteSummaryResponse();
        summary.setDistance(distance);
        summary.setDuration(durationSec);
        summary.setTaxiFare(raw.getTaxiFare());
        summary.setTollFare(raw.getTollFare());

        if (raw.getPath() != null) {
            summary.setPath(
                    raw.getPath().stream()
                            .map(p -> new LatLngDto(p.getLat(), p.getLng()))
                            .collect(Collectors.toList())
            );
        }

        return summary;
    }
}
