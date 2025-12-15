package org.iclass;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.core.publisher.Mono;

@RestController
public class FastAPIProxyController {

    // 1. Spring 설정 파일(application.yml/properties)에서 FastAPI의 실제 Base URL을 로드합니다.
    //    예: stats.fastapi.base-url=http://finalproject-production-c135.up.railway.app
    @Value("${stats.fastapi.base-url}") 
    private String fastApiBaseUrl; 

    // 2. FastAPI 서버의 실제 Host 이름을 상수로 정의합니다.
    private static final String RAILWAY_FASTAPI_HOST = "finalproject-production-c135.up.railway.app";
    
    private final WebClient webClient;

    public FastAPIProxyController(WebClient.Builder webClientBuilder) {
        // webClient를 초기화할 때 Base URL을 사용합니다.
        // WebClient는 fastApiBaseUrl로 요청을 보냅니다.
        this.webClient = webClientBuilder.baseUrl(fastApiBaseUrl).build(); 
    }

    // summary 요청 프록시 (GET)
    // 프론트엔드 요청: /api/stores/{storeId}/summary
    @GetMapping("/api/stores/{storeId}/summary")
    public Mono<ResponseEntity<String>> proxySummary(@PathVariable String storeId) {
        String fastApiPath = "/api/stores/" + storeId + "/summary"; 
        
        return webClient.get()
                .uri(fastApiPath)
                // 3. FastAPI가 구동 중인 Railway의 Host 헤더를 명시적으로 설정합니다.
                .header("Host", RAILWAY_FASTAPI_HOST)
                .retrieve()
                .toEntity(String.class);
    }

    // recommend 요청 프록시 (POST)
    // 프론트엔드 요청: /recommend/route
    @PostMapping("/recommend/route")
    public Mono<ResponseEntity<String>> proxyRecommend(@RequestBody String requestBody) {
        String fastApiPath = "/recommend/route"; 

        return webClient.post()
                .uri(fastApiPath)
                // 3. FastAPI가 구동 중인 Railway의 Host 헤더를 명시적으로 설정합니다.
                .header("Host", RAILWAY_FASTAPI_HOST)
                .bodyValue(requestBody)
                .retrieve()
                .toEntity(String.class);
    }
}