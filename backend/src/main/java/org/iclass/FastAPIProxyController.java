package org.iclass;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.core.publisher.Mono; // 🚨 필수 임포트 (Reactive Streams)

@RestController
@RequestMapping("/api") // /api 하위 경로는 Spring Boot 서버가 처리해야 합니다.
public class FastAPIProxyController {

    // application.properties에서 설정된 FastAPI 서버 URL을 주입받습니다.
    @Value("${stats.fastapi.base-url}") 
    private String fastApiBaseUrl; 

    private final WebClient webClient;

    public FastAPIProxyController(WebClient.Builder webClientBuilder) {
        // WebClient 인스턴스를 생성하며, FastAPI 기본 URL을 설정합니다.
        // 현재는 WebClient.Builder 주입 방식을 그대로 사용합니다.
        this.webClient = webClientBuilder.baseUrl(fastApiBaseUrl).build(); 
    }

    /**
     * [GET] /api/stores/{storeId}/summary 요청을 FastAPI로 포워딩합니다.
     * 반환 타입 Mono를 사용하여 비동기(Non-Blocking) 방식으로 응답합니다.
     */
    @GetMapping("/stores/{storeId}/summary")
    public Mono<ResponseEntity<String>> proxySummary(@PathVariable String storeId) {
        String fastApiPath = "/api/stores/" + storeId + "/summary"; // FastAPI의 실제 경로

        return webClient.get()
                .uri(fastApiPath)
                .retrieve()
                .toEntity(String.class); // Mono<ResponseEntity<String>> 반환 (비동기)
    }

    /**
     * [POST] /recommend/route 요청을 FastAPI로 포워딩합니다.
     * 경로 추천 요청은 /api 경로 아래에 있지 않으므로 @RequestMapping과 별개로 매핑합니다.
     * 반환 타입 Mono를 사용하여 비동기(Non-Blocking) 방식으로 응답합니다.
     */
    @PostMapping("/recommend/route")
    public Mono<ResponseEntity<String>> proxyRecommend(@RequestBody String requestBody) {
        String fastApiPath = "/recommend/route"; // FastAPI의 실제 경로

        return webClient.post()
                .uri(fastApiPath)
                .bodyValue(requestBody) // 요청 본문을 그대로 전달
                .retrieve()
                .toEntity(String.class); // Mono<ResponseEntity<String>> 반환 (비동기)
    }
}