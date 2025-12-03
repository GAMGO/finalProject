// src/main/java/org/iclass/route/service/KakaoRouteService.java
package org.iclass.route.service;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import io.github.cdimascio.dotenv.Dotenv;
import org.iclass.route.dto.LatLngDto;
import org.iclass.route.dto.RoutePoint;
import org.iclass.route.dto.RouteResponse;
import org.iclass.route.dto.RouteSummaryResponse;
import org.iclass.route.dto.TransportMode;
import org.springframework.http.*;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;
import org.springframework.web.util.UriComponentsBuilder;

import java.util.ArrayList;
import java.util.List;

@Service
public class KakaoRouteService {

    private final RestTemplate restTemplate;
    private final ObjectMapper objectMapper;
    private final String kakaoApiKey;

    private static final String BASE_URL = "https://apis-navi.kakaomobility.com/v1/directions";

    public KakaoRouteService() {
        this.restTemplate = new RestTemplate();
        this.objectMapper = new ObjectMapper();

        String key = System.getenv("KAKAO_REST_API_KEY");

        if (key == null || key.isBlank()) {
            try {
                Dotenv dotenv = Dotenv.configure()
                        .ignoreIfMalformed()
                        .ignoreIfMissing()
                        .load();
                String fromEnv = dotenv.get("KAKAO_REST_API_KEY");
                if (fromEnv != null && !fromEnv.isBlank()) {
                    key = fromEnv;
                }
            } catch (Exception ignore) {
            }
        }

        if (key == null || key.isBlank()) {
            throw new IllegalStateException(
                    "KAKAO_REST_API_KEY 환경변수를 찾을 수 없습니다. " +
                            "OS 환경변수나 .env에 설정해 주세요.");
        }

        this.kakaoApiKey = key;
    }

    /**
     * 카카오 내비 원본 경로 정보 받아오기 (차량 기준)
     */
    public RouteResponse getRoute(double originLat,
            double originLng,
            double destLat,
            double destLng) {

        // 카카오 내비는 "경도,위도" (lng,lat)
        String originParam = originLng + "," + originLat;
        String destParam = destLng + "," + destLat;

        String url = UriComponentsBuilder
                .fromHttpUrl(BASE_URL)
                .queryParam("origin", originParam)
                .queryParam("destination", destParam)
                .queryParam("priority", "RECOMMEND")
                .toUriString();

        HttpHeaders headers = new HttpHeaders();
        headers.set("Authorization", "KakaoAK " + kakaoApiKey);
        headers.setContentType(MediaType.APPLICATION_JSON);

        HttpEntity<Void> entity = new HttpEntity<>(headers);

        ResponseEntity<String> response = restTemplate.exchange(
                url,
                HttpMethod.GET,
                entity,
                String.class);

        if (!response.getStatusCode().is2xxSuccessful()) {
            throw new IllegalStateException(
                    "Kakao directions API 호출 실패: " + response.getStatusCode());
        }

        try {
            String body = response.getBody();
            JsonNode root = objectMapper.readTree(body);

            JsonNode routes = root.path("routes");
            if (!routes.isArray() || routes.isEmpty()) {
                throw new IllegalStateException("경로 정보를 찾을 수 없습니다.");
            }

            JsonNode firstRoute = routes.get(0);
            JsonNode summary = firstRoute.path("summary");

            double distance = summary.path("distance").asDouble(0); // m
            double duration = summary.path("duration").asDouble(0); // sec
            Integer taxiFare = null;
            JsonNode fareNode = summary.path("fare");
            if (fareNode.has("taxi")) {
                taxiFare = fareNode.path("taxi").asInt();
            }

            // ✅ vertexes 파싱 (여러 구조 대응)
            List<RoutePoint> path = new ArrayList<>();

            JsonNode sections = firstRoute.path("sections");
            if (sections.isArray()) {
                for (JsonNode section : sections) {

                    boolean addedFromRoads = false;

                    // 1) sections[*].roads[*].vertexes 구조
                    JsonNode roads = section.path("roads");
                    if (roads.isArray() && roads.size() > 0) {
                        for (JsonNode road : roads) {
                            JsonNode vertexes = road.path("vertexes");
                            if (vertexes.isArray()) {
                                for (int i = 0; i + 1 < vertexes.size(); i += 2) {
                                    double lng = vertexes.get(i).asDouble();
                                    double lat = vertexes.get(i + 1).asDouble();
                                    path.add(new RoutePoint(lat, lng));
                                }
                            }
                        }
                        addedFromRoads = true;
                    }

                    // 2) sections[*].vertexes 구조 (fallback)
                    if (!addedFromRoads) {
                        JsonNode vertexes = section.path("vertexes");
                        if (vertexes.isArray()) {
                            for (int i = 0; i + 1 < vertexes.size(); i += 2) {
                                double lng = vertexes.get(i).asDouble();
                                double lat = vertexes.get(i + 1).asDouble();
                                path.add(new RoutePoint(lat, lng));
                            }
                        }
                    }
                }
            } else {
                // 3) routes[0].vertexes 구조 (추가 fallback)
                JsonNode vertexes = firstRoute.path("vertexes");
                if (vertexes.isArray()) {
                    for (int i = 0; i + 1 < vertexes.size(); i += 2) {
                        double lng = vertexes.get(i).asDouble();
                        double lat = vertexes.get(i + 1).asDouble();
                        path.add(new RoutePoint(lat, lng));
                    }
                }
            }

            RouteResponse dto = new RouteResponse();
            dto.setDistance(distance);
            dto.setDuration(duration);
            dto.setTaxiFare(taxiFare);
            dto.setPath(path);

            return dto;
        } catch (Exception e) {
            throw new IllegalStateException("Kakao directions 응답 파싱 실패", e);
        }
    }

    /**
     * 프론트에서 쓰기 좋은 형태로 변환해서 리턴
     *
     * ※ 지금은 카카오 내비 API가 차량 기준이라,
     * 도보/대중교통 모드는 자동차 경로를 기반으로
     * 예상 시간만 다르게 계산해서 내려줌.
     * (실제 서비스에서는 전용 API로 교체 추천)
     */
    public RouteSummaryResponse searchRoute(TransportMode mode,
            double originLat,
            double originLng,
            double destLat,
            double destLng) {

        if (mode == null) {
            mode = TransportMode.CAR;
        }

        RouteResponse raw = getRoute(originLat, originLng, destLat, destLng);

        int distance = (int) Math.round(raw.getDistance()); // m
        int baseDuration = (int) Math.round(raw.getDuration()); // sec
        int durationSec;
        switch (mode) {
            case WALK:
                double walkSpeed = 1.3;
                durationSec = (int) Math.round(distance / walkSpeed);
                break;
            case TRANSIT:
                durationSec = baseDuration + 5 * 60;
                break;
            case CAR:
            default:
                durationSec = baseDuration;
                break;
        }

        RouteSummaryResponse summary = new RouteSummaryResponse();
        summary.setDistance(distance);
        summary.setDuration(durationSec);
        summary.setTaxiFare(raw.getTaxiFare());
        summary.setTollFare(null); // 유료도로 요금은 현재 사용 안 함

        if (raw.getPath() != null) {
            List<LatLngDto> path = raw.getPath().stream()
                    .map(p -> new LatLngDto(p.getLat(), p.getLng()))
                    .toList();
            summary.setPath(path);
        }

        return summary;
    }
}
