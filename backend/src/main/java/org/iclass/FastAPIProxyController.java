package org.iclass;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.core.publisher.Mono;

@RestController
public class FastAPIProxyController {

    @Value("${stats.fastapi.base-url}") 
    private String fastApiBaseUrl; 

    private final WebClient webClient;

    public FastAPIProxyController(WebClient.Builder webClientBuilder) {
        this.webClient = webClientBuilder.baseUrl(fastApiBaseUrl).build(); 
    }

    // summary 요청 프록시
    @GetMapping("/api/stores/{storeId}/summary")
    public Mono<ResponseEntity<String>> proxySummary(@PathVariable String storeId) {
        String fastApiPath = "/api/stores/" + storeId + "/summary"; 
        
        return webClient.get()
                .uri(fastApiPath)
                .header("Host", "finalproject-railway.app")
                .retrieve()
                .toEntity(String.class);
    }

    // recommend 요청 프록시
    @PostMapping("/recommend/route")
    public Mono<ResponseEntity<String>> proxyRecommend(@RequestBody String requestBody) {
        String fastApiPath = "/recommend/route"; 

        return webClient.post()
                .uri(fastApiPath)
                .header("Host", "finalproject-railway.app")
                .bodyValue(requestBody)
                .retrieve()
                .toEntity(String.class);
    }
}