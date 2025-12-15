package org.iclass;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.core.publisher.Mono;

@RestController
public class FastAPIProxyController {

    // application.properties에서 설정된 FastAPI 서버 URL을 주입받습니다.
    @Value("${stats.fastapi.base-url}") 
    private String fastApiBaseUrl; 

    private final WebClient webClient;

    public FastAPIProxyController(WebClient.Builder webClientBuilder) {
        // WebClient 인스턴스를 생성하며, FastAPI 기본 URL을 설정합니다.
        // 이 인스턴스는 한 번 생성된 후 재사용됩니다.
        this.webClient = webClientBuilder.baseUrl(fastApiBaseUrl).build(); 
    }

    /**
     * [GET] /api/stores/{storeId}/summary 요청을 FastAPI로 포워딩합니다.
     * Host 헤더를 명시적으로 추가하여 FastAPI가 올바른 도메인(finalproject-railway.app)을 인식하도록 합니다.
     */
    @GetMapping("/api/stores/{storeId}/summary")
    public Mono<ResponseEntity<String>> proxySummary(@PathVariable String storeId) {
        String fastApiPath = "/api/stores/" + storeId + "/summary"; 
        
        return webClient.get()
                .uri(fastApiPath)
                .header("Host", "finalproject-railway.app") // 🚨 핵심: Host 헤더 강제 지정
                .retrieve()
                .toEntity(String.class); // Mono<ResponseEntity<String>> 반환 (비동기)
    }

    /**
     * [POST] /recommend/route 요청을 FastAPI로 포워딩합니다.
     * Host 헤더를 명시적으로 추가하여 FastAPI가 올바른 도메인(finalproject-railway.app)을 인식하도록 합니다.
     */
    @PostMapping("/recommend/route")
    public Mono<ResponseEntity<String>> proxyRecommend(@RequestBody String requestBody) {
        String fastApiPath = "/recommend/route"; 

        return webClient.post()
                .uri(fastApiPath)
                .header("Host", "finalproject-railway.app") // 🚨 핵심: Host 헤더 강제 지정
                .bodyValue(requestBody) // 요청 본문을 그대로 전달
                .retrieve()
                .toEntity(String.class); // Mono<ResponseEntity<String>> 반환 (비동기)
    }
}