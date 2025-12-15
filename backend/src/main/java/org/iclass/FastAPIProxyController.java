package org.iclass;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.reactive.function.client.WebClient;

@RestController
public class FastAPIProxyController {

    // application.properties의 VITE_BASE_URL (Railway 주소)를 주입받습니다.
    @Value("${VITE_BASE_URL}") 
    private String fastApiBaseUrl; 

    private final WebClient webClient;

    public FastAPIProxyController(WebClient.Builder webClientBuilder) {
        // 기본 WebClient 설정
        this.webClient = webClientBuilder.baseUrl(fastApiBaseUrl).build(); 
    }

    // 1. '/api/stores/{storeId}/summary' 경로 프록시 로직
    @GetMapping("/api/stores/{storeId}/summary")
    public ResponseEntity<String> proxySummary(@PathVariable String storeId) {
        // Railway FastAPI 서버의 전체 경로를 만듭니다. (FastAPI의 라우팅 prefix를 포함)
        String fastApiPath = "/api/stores/" + storeId + "/summary";

        // WebClient를 사용하여 Railway로 요청을 보내고 응답을 그대로 반환합니다.
        return webClient.get()
                .uri(fastApiPath)
                .retrieve()
                .toEntity(String.class)
                .block(); // 비동기 WebClient를 동기적으로 처리 (간단한 예시)
    }

    // 2. '/recommend/route' 경로 프록시 로직
    @PostMapping("/recommend/route") // 일반적으로 POST일 가능성이 높습니다.
    public ResponseEntity<String> proxyRecommend(@RequestBody String requestBody) {
        String fastApiPath = "/recommend/route";

        return webClient.post()
                .uri(fastApiPath)
                .bodyValue(requestBody)
                .retrieve()
                .toEntity(String.class)
                .block(); 
    }
}
