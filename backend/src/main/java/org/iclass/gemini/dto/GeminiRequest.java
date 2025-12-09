// src/main/java/org/iclass/gemini/dto/GeminiRequest.java
package org.iclass.gemini.dto;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.List;

/**
 * Gemini generateContent 요청 형식:
 * {
 *   "contents": [
 *     {
 *       "parts": [ { "text": "..." } ]
 *     }
 *   ]
 * }
 */
@Data
@NoArgsConstructor
@AllArgsConstructor
public class GeminiRequest {

    private List<Content> contents;

    // 편하게 쓰려고 prompt 하나만 받는 생성자
    public GeminiRequest(String prompt) {
        this.contents = List.of(new Content(List.of(new Part(prompt))));
    }

    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    public static class Content {
        private List<Part> parts;
    }

    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    public static class Part {
        private String text;
    }
}
