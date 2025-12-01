// src/main/java/org/iclass/route/service/KakaoRouteService.java
package org.iclass.route.service;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import io.github.cdimascio.dotenv.Dotenv;
import org.iclass.route.dto.RoutePoint;
import org.iclass.route.dto.RouteResponse;
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

    /**
     * ✅ 기본 생성자 (Spring이 이걸로 빈 생성함)
     * - KAKAO_REST_API_KEY는
     *   1) 시스템 환경변수
     *   2) .env (java-dotenv)
     *   두 군데 중 하나에서 가져옴.
     */
    public KakaoRouteService() {
        this.restTemplate = new RestTemplate();
        this.objectMapper = new ObjectMapper();

        // 1) OS 환경변수에서 먼저 찾기
        String key = System.getenv("KAKAO_REST_API_KEY");

        // 2) 못 찾으면 .env에서 찾기
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
                // .env가 없거나 해도 그냥 무시
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
     * 길찾기 요청
     * @param originLat  출발지 위도
     * @param originLng  출발지 경도
     * @param destLat    도착지 위도
     * @param destLng    도착지 경도
     */
    public RouteResponse getRoute(double originLat,
                                  double originLng,
                                  double destLat,
                                  double destLng) {

        // 카카오 내비는 "경도,위도" 순서
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

            // vertexes: [lng1, lat1, lng2, lat2, ...]
            List<RoutePoint> path = new ArrayList<>();
            JsonNode sections = firstRoute.path("sections");
            for (JsonNode section : sections) {
                JsonNode vertexes = section.path("vertexes");
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
}
