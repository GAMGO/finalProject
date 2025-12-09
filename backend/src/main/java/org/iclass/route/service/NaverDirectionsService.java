// src/main/java/org/iclass/route/service/NaverDirectionsService.java
package org.iclass.route.service;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.extern.slf4j.Slf4j;
import org.iclass.route.dto.LatLngDto;
import org.iclass.route.dto.RoutePoint;
import org.iclass.route.dto.RouteResponse;
import org.iclass.route.dto.TransportMode;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.*;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;
import org.springframework.web.util.UriComponentsBuilder;

import java.util.ArrayList;
import java.util.List;

@Service
@Slf4j
public class NaverDirectionsService {

    @Value("${naver.maps.client-id}")
    private String clientId;

    @Value("${naver.maps.client-secret}")
    private String clientSecret;

    private final RestTemplate restTemplate = new RestTemplate();
    private final ObjectMapper objectMapper = new ObjectMapper();

    // 네이버 Directions5 driving 엔드포인트
    private static final String BASE_URL =
            "https://naveropenapi.apigw.ntruss.com/map-direction/v1/driving";

    /**
     * 네이버 Directions 호출
     */
    public RouteResponse getRoute(LatLngDto from, LatLngDto to, TransportMode mode) {

        // 네이버는 "경도,위도" = lng,lat 순서
        String start = from.getLng() + "," + from.getLat();
        String goal  = to.getLng() + "," + to.getLat();

        // option 값 + route JSON 안에서의 키가 같음 (trafast, tracomfort 등)
        String option = convertMode(mode);

        String url = UriComponentsBuilder.fromHttpUrl(BASE_URL)
                .queryParam("start", start)
                .queryParam("goal", goal)
                .queryParam("option", option)
                .toUriString();

        HttpHeaders headers = new HttpHeaders();
        headers.set("X-NCP-APIGW-API-KEY-ID", clientId);
        headers.set("X-NCP-APIGW-API-KEY", clientSecret);

        HttpEntity<Void> entity = new HttpEntity<>(headers);

        log.info("[NAVER] 요청 URL = {}", url);

        ResponseEntity<String> res = restTemplate.exchange(
                url,
                HttpMethod.GET,
                entity,
                String.class
        );

        log.info("[NAVER] status={} body={}", res.getStatusCode(), res.getBody());

        if (!res.getStatusCode().is2xxSuccessful()) {
            throw new IllegalStateException("Naver Directions API 에러: " + res.getStatusCode());
        }

        try {
            String body = res.getBody();
            JsonNode root = objectMapper.readTree(body);

            int code = root.path("code").asInt(-1);
            String message = root.path("message").asText();

            if (code != 0) {
                throw new IllegalStateException(
                        "Naver Directions 에러 code=" + code + " msg=" + message);
            }

            // route.trafast[0] 이런 구조
            JsonNode routeArray = root.path("route").path(option);
            if (!routeArray.isArray() || routeArray.isEmpty()) {
                throw new IllegalStateException("경로 정보가 없습니다.");
            }

            JsonNode firstRoute = routeArray.get(0);

            // ===== summary 파싱 (거리/시간/요금) =====
            JsonNode summary = firstRoute.path("summary");
            double distance = summary.path("distance").asDouble(0);  // meter
            double duration = summary.path("duration").asDouble(0);  // ms (문서 기준)

            Integer taxiFare = summary.has("taxiFare") && !summary.get("taxiFare").isNull()
                    ? summary.get("taxiFare").asInt()
                    : null;

            Integer tollFare = summary.has("tollFare") && !summary.get("tollFare").isNull()
                    ? summary.get("tollFare").asInt()
                    : null;

            // ===== path 파싱 (경로 좌표들) =====
            List<RoutePoint> points = new ArrayList<>();
            JsonNode pathNode = firstRoute.path("path"); // [[lng,lat], [lng,lat], ...]

            if (pathNode.isArray()) {
                for (JsonNode pair : pathNode) {
                    if (pair.isArray() && pair.size() >= 2) {
                        double lng = pair.get(0).asDouble();
                        double lat = pair.get(1).asDouble();
                        points.add(new RoutePoint(lat, lng));
                    }
                }
            }

            // ===== DTO 만들기 =====
            RouteResponse dto = new RouteResponse();
            dto.setDistance(distance);
            dto.setDuration(duration / 1000.0); // ms → 초로 바꾸고 싶으면 이렇게
            dto.setTaxiFare(taxiFare);
            dto.setTollFare(tollFare);
            dto.setPath(points);

            return dto;

        } catch (Exception e) {
            throw new IllegalStateException("Naver Directions 응답 파싱 실패", e);
        }
    }

    /**
     * 우리 enum TransportMode → 네이버 option 값 매핑
     * (일단 자동차용 엔드포인트 하나 쓰면서 옵션만 바꾸는 구조)
     */
    private String convertMode(TransportMode mode) {
        if (mode == null) return "trafast";

        switch (mode) {
            case WALK:
                // 실제로는 전용 도보 API 있으면 그걸로 교체.
                // 지금은 자동차 최단경로 기준으로만 경로 뽑고,
                // 소요시간은 나중에 별도로 계산하는 방식으로 써도 됨.
                return "trafast";
            case TRANSIT:
                // 대중교통도 아직 Directions 5에서 완전히 지원 안 하니까
                // 일단 동일 경로 + 다른 속도 계산용으로 trafast만 사용.
                return "trafast";
            case CAR:
            default:
                return "trafast";   // 자동차 최단 시간
        }
    }
}
