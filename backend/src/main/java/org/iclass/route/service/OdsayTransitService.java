package org.iclass.route.service;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import io.github.cdimascio.dotenv.Dotenv;
import org.iclass.route.dto.LatLngDto;
import org.iclass.route.dto.RouteSummaryResponse;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpMethod;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;
import org.springframework.web.util.UriComponentsBuilder;

import java.util.ArrayList;
import java.util.List;

/**
 * ODsay 대중교통 길찾기 서비스
 *  - https://api.odsay.com/v1/api/searchPubTransPathT 사용
 */
@Service
public class OdsayTransitService {

    private static final Logger log = LoggerFactory.getLogger(OdsayTransitService.class);

    private final RestTemplate restTemplate;
    private final ObjectMapper objectMapper;
    private final String odsayApiKey;

    private static final String BASE_URL =
            "https://api.odsay.com/v1/api/searchPubTransPathT";

    public OdsayTransitService() {
        this.restTemplate = new RestTemplate();
        this.objectMapper = new ObjectMapper();

        // ✅ 환경변수 또는 .env 에서 키 읽기 (KakaoRouteService 와 동일 패턴)
        String key = System.getenv("ODSAY_API_KEY");

        if (key == null || key.isBlank()) {
            try {
                Dotenv dotenv = Dotenv.configure()
                        .ignoreIfMalformed()
                        .ignoreIfMissing()
                        .load();
                String fromEnv = dotenv.get("ODSAY_API_KEY");
                if (fromEnv != null && !fromEnv.isBlank()) {
                    key = fromEnv;
                }
            } catch (Exception ignore) {
            }
        }

        if (key == null || key.isBlank()) {
            throw new IllegalStateException(
                    "ODSAY_API_KEY 환경변수를 찾을 수 없습니다. " +
                    ".env 또는 OS 환경변수에 ODSAY_API_KEY를 설정해 주세요.");
        }

        this.odsayApiKey = key;
    }

    /**
     * 대중교통 경로 검색
     *  - originLat/originLng : 출발지
     *  - destLat/destLng     : 도착지
     */
    public RouteSummaryResponse searchTransitRoute(
            double originLat, double originLng,
            double destLat, double destLng
    ) {

        // ODsay는 SX,SY = 출발 경도/위도, EX,EY = 도착 경도/위도
        String url = UriComponentsBuilder.fromHttpUrl(BASE_URL)
                .queryParam("SX", originLng)
                .queryParam("SY", originLat)
                .queryParam("EX", destLng)
                .queryParam("EY", destLat)
                .queryParam("apiKey", odsayApiKey)
                .toUriString();

        log.info("[ODsay] 요청 URL = {}", url);

        ResponseEntity<String> res = restTemplate.exchange(
                url,
                HttpMethod.GET,
                new HttpEntity<>(null),
                String.class
        );

        String body = res.getBody();
        log.info("[ODsay] status={}, body={}", res.getStatusCode(), body);

        if (!res.getStatusCode().is2xxSuccessful() || body == null) {
            throw new IllegalStateException("ODsay API 호출 실패: " + res.getStatusCode());
        }

        try {
            JsonNode root = objectMapper.readTree(body);

            // 1) error 필드 먼저 체크 (키 오류, 일일 호출 초과 등)
            JsonNode errorNode = root.path("error");
            if (!errorNode.isMissingNode() && !errorNode.isNull()) {
                String code = errorNode.path("code").asText();
                String msg = errorNode.path("msg").asText();
                throw new IllegalStateException("ODsay API 에러: code=" + code + ", msg=" + msg);
            }

            // 2) result/path 배열 안전하게 꺼내기
            JsonNode resultNode = root.path("result");
            if (resultNode.isMissingNode() || resultNode.isNull()) {
                log.error("[ODsay] result 노드 없음. raw={}", body);
                throw new IllegalStateException("ODsay result 데이터 없음");
            }

            JsonNode pathArray = resultNode.path("path");
            if (pathArray.isMissingNode() || !pathArray.isArray() || pathArray.size() == 0) {
                log.error("[ODsay] path 배열 없음 또는 비어있음. raw={}", body);
                throw new IllegalStateException("ODsay 경로 데이터 없음");
            }

            JsonNode path0 = pathArray.get(0);
            if (path0 == null || path0.isNull()) {
                log.error("[ODsay] path[0] 없음. raw={}", body);
                throw new IllegalStateException("ODsay 경로 데이터 없음(path[0])");
            }

            JsonNode info = path0.path("info");

            int totalTimeMin = info.path("totalTime").asInt(0);      // 분
            int totalDistance = info.path("totalDistance").asInt(0); // m
            Integer payment = info.hasNonNull("payment")
                    ? info.path("payment").asInt()
                    : null;

            // 3) 지나는 정류장 좌표를 path 로 사용
            List<LatLngDto> pathPoints = new ArrayList<>();

            JsonNode subPathArr = path0.path("subPath");
            if (subPathArr.isArray()) {
                for (JsonNode sp : subPathArr) {
                    JsonNode stations = sp.path("passStopList").path("stations");
                    if (!stations.isArray()) continue;

                    for (JsonNode st : stations) {
                        double lat = st.path("y").asDouble(); // 위도
                        double lng = st.path("x").asDouble(); // 경도
                        pathPoints.add(new LatLngDto(lat, lng));
                    }
                }
            }

            // 혹시 정류장 좌표가 비어 있으면 최소한 출발/도착만 넣어줌
            if (pathPoints.isEmpty()) {
                log.warn("[ODsay] 좌표 데이터 없음. 출발/도착만 사용. raw={}", body);
                pathPoints.add(new LatLngDto(originLat, originLng));
                pathPoints.add(new LatLngDto(destLat, destLng));
            }

            RouteSummaryResponse summary = new RouteSummaryResponse();
            summary.setDistance(totalDistance);
            summary.setDuration(totalTimeMin * 60);   // 초 단위로 변환
            summary.setTaxiFare(null);                // 대중교통에서는 사용 안 함
            summary.setTollFare(payment);             // 편하게 요금 넣어둠
            summary.setPath(pathPoints);

            return summary;
        } catch (Exception e) {
            log.error("[ODsay] 응답 파싱 실패", e);
            throw new IllegalStateException("ODsay 응답 파싱 실패", e);
        }
    }
}
