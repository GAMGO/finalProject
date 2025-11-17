package com.finalproject.backend.service;

import lombok.Data;
import lombok.RequiredArgsConstructor;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.*;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;

import java.util.List;
import java.util.Map;

@Service
@RequiredArgsConstructor
public class OpenAiClient {

    private final RestTemplate restTemplate;

    @Value("${openai.api.url}")
    private String apiUrl;

    @Value("${openai.model}")
    private String model;

    @Value("${openai.api.key}")
    private String apiKey;

    @Data
    public static class OpenAiChatResponse {
        private List<Choice> choices;

        @Data
        public static class Choice {
            private Message message;
        }

        @Data
        public static class Message {
            private String role;
            private String content;
        }
    }

    public String chat(List<Map<String, String>> messages) {
        Map<String, Object> body = Map.of(
                "model", model,
                "messages", messages
        );

        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);
        headers.setBearerAuth(apiKey);

        HttpEntity<Map<String, Object>> entity = new HttpEntity<>(body, headers);

        ResponseEntity<OpenAiChatResponse> res = restTemplate.exchange(
                apiUrl,
                HttpMethod.POST,
                entity,
                OpenAiChatResponse.class
        );

        OpenAiChatResponse response = res.getBody();
        if (response == null || response.getChoices() == null || response.getChoices().isEmpty()) {
            return "AI 응답이 없습니다.";
        }
        return response.getChoices().get(0).getMessage().getContent();
    }
}
