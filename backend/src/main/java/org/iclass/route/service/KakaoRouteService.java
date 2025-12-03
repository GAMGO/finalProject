// src/main/java/org/iclass/route/service/KakaoRouteService.java
package org.iclass.route.service;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import io.github.cdimascio.dotenv.Dotenv;
import org.iclass.route.dto.LatLngDto;
import org.iclass.route.dto.RoutePoint;
import org.iclass.route.dto.RouteResponse;
import org.iclass.route.dto.RouteSummaryResponse;
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
                    "OS 환경변수나 .env에 설정해 주세요."
            );
        }

        this.kakaoApiKey = key;
    }

    /**
     * 카카오 내비 원본 경로 정보 받아오기
     */
    public RouteResponse getRoute(double originLat,
                                  double originLng,
                                  double destLat,
                                  double destLng) {

        // 카카오 내비는 "경도,위도" (lng,lat)
        String originParam = originLng + "," + originLat;
        String destParam   = destLng + "," + destLat;

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
                String.class
        );

        if (!response.getStatusCode().is2xxSuccessful()) {
            throw new IllegalStateException(
                    "Kakao directions API 호출 실패: " + response.getStatusCode()
            );
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

            // ✅ vertexes 파싱: routes[0].sections[*].roads[*].vertexes
            List<RoutePoint> path = new ArrayList<>();
            JsonNode sections = firstRoute.path("sections");
            if (sections.isArray()) {
                for (JsonNode section : sections) {
                    JsonNode roads = section.path("roads");
                    if (!roads.isArray()) continue;

                    for (JsonNode road : roads) {
                        JsonNode vertexes = road.path("vertexes");
                        if (!vertexes.isArray()) continue;

                        // [lng1, lat1, lng2, lat2, ...]
                        for (int i = 0; i + 1 < vertexes.size(); i += 2) {
                            double lng = vertexes.get(i).asDouble();
                            double lat = vertexes.get(i + 1).asDouble();
                            path.add(new RoutePoint(lat, lng));
                        }
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
     */
    public RouteSummaryResponse searchRoute(double originLat,
                                            double originLng,
                                            double destLat,
                                            double destLng) {

        RouteResponse raw = getRoute(originLat, originLng, destLat, destLng);

        RouteSummaryResponse summary = new RouteSummaryResponse();
        summary.setDistance((int) Math.round(raw.getDistance()));   // m
        summary.setDuration((int) Math.round(raw.getDuration()));   // sec
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
