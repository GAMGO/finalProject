// src/main/java/org/iclass/gemini/ReviewModerationService.java
package org.iclass.gemini;

import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.iclass.gemini.dto.GeminiRequest;
import org.iclass.gemini.dto.GeminiResponse;
import org.iclass.gemini.dto.ModerationResult;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;

import java.util.Optional;

@Slf4j
@Service
@RequiredArgsConstructor
public class ReviewModerationService {

    private final RestTemplate geminiRestTemplate;   // GeminiConfig에서 만든 Bean
    private final ObjectMapper objectMapper;         // Spring Boot가 기본 제공

    @Value("${gemini.api.url}")
    private String geminiApiUrl;

    /**
     * 리뷰 텍스트를 전달하면
     *  - ALLOW : 통과
     *  - BLOCK : 욕설/비하 등으로 차단
     *  - REVIEW: 사람이 한 번 더 보면 좋겠음 (지금은 ALLOW처럼 처리)
     */
    public ModerationResult moderate(String reviewText) {
        log.info("[MOD] input review = {}", reviewText);

        // 공백이면 그냥 허용
        if (reviewText == null || reviewText.trim().isEmpty()) {
            ModerationResult r = new ModerationResult();
            r.setDecision("ALLOW");
            r.setReason("empty_or_whitespace");
            return r;
        }

        String prompt = buildPrompt(reviewText);
        GeminiRequest request = new GeminiRequest(prompt);

        try {
            GeminiResponse response =
                    geminiRestTemplate.postForObject(geminiApiUrl, request, GeminiResponse.class);

            String rawText = extractFirstText(response);
            log.info("[MOD] rawText from Gemini: {}", rawText);

            if (rawText == null || rawText.isBlank()) {
                ModerationResult r = new ModerationResult();
                r.setDecision("ALLOW");
                r.setReason("empty_response");
                return r;
            }

            // ✂ Gemini가 앞뒤에 잡소리를 붙여도, 중간의 JSON만 잘라서 파싱
            String json = extractJson(rawText);

            // JSON 파싱
            ModerationResult result =
                    objectMapper.readValue(json, ModerationResult.class);

            if (result.getDecision() == null) {
                result.setDecision("ALLOW");
            }

            log.info("[MOD] parsed moderation result: {}", result);
            return result;
        } catch (Exception e) {
            // ❗ 테스트 동안엔 에러 나면 무조건 막아버리기
            log.error("Gemini moderation error, BLOCK for safety", e);
            ModerationResult r = new ModerationResult();
            r.setDecision("BLOCK");
            r.setReason("gemini_error_block");
            return r;
        }
    }

    /**
     * Gemini 응답에서 맨 첫 번째 text만 추출
     */
    private String extractFirstText(GeminiResponse response) {
        if (response == null) return null;
        return Optional.ofNullable(response.getCandidates())
                .flatMap(cands -> cands.stream().findFirst())
                .map(GeminiResponse.Candidate::getContent)
                .flatMap(content -> Optional.ofNullable(content.getParts())
                        .flatMap(parts -> parts.stream().findFirst()))
                .map(GeminiResponse.Part::getText)
                .orElse(null);
    }

    /**
     * Gemini가 ```json ... ``` 이나 앞뒤 설명이 섞여 있어도
     * 중간의 { ... } JSON 객체만 뽑아낸다.
     */
    private String extractJson(String rawText) {
        if (rawText == null) {
            throw new IllegalArgumentException("rawText is null");
        }

        int start = rawText.indexOf('{');
        int end = rawText.lastIndexOf('}');
        if (start >= 0 && end > start) {
            return rawText.substring(start, end + 1);
        }
        throw new IllegalArgumentException("No JSON object found in: " + rawText);
    }

    /**
     * 모더레이션 프롬프트
     */
    private String buildPrompt(String reviewText) {
        return """
                너는 길거리 음식 리뷰 앱의 콘텐츠 모더레이션 시스템이야.
                사용자가 남긴 한국어 리뷰가 다음 기준에 해당하는지 판단해.

                [BLOCK 기준]
                - 욕설, 심한 비속어, 성적 모욕, 인종/지역/성별/장애 등 혐오 발언
                - 가게 주인/직원/특정 손님을 향한 인신 공격, 모욕, 비하
                - 고의적인 도배, 광고, 스팸(전화번호, 링크 등 홍보 목적)
                - 협박, 폭력 선동, 스토킹성 발언

                [ALLOW 기준]
                - 음식이 맛없다고 솔직하게 불평하는 리뷰 (예: "진짜 맛없음", "두 번은 안 시킬 듯")
                - 서비스가 불친절했다는 비판 (예: "사장님이 좀 불친절했어요")
                - 욕설 없이 강한 표현, 감정 표현

                [REVIEW 기준]
                - 애매해서 자동으로 BLOCK하기 애매한 경우
                - 농담인지 모욕인지 판단이 애매한 표현

                [예시]
                입력: "야 이 장애인 같은 새끼야 가게 접어라"
                출력: {"decision":"BLOCK","reason":"장애인 비하와 심한 욕설"}

                입력: "맛없고 다시는 안 올 거 같아요"
                출력: {"decision":"ALLOW","reason":"욕설 없는 부정적 리뷰"}

                출력 형식:
                JSON 객체 하나만 반환해.
                마크다운 백틱(```) 쓰지 말고,
                다른 설명도 쓰지 말고,
                오직 아래 형식의 JSON만 응답해.

                {
                  "decision": "ALLOW" 또는 "BLOCK" 또는 "REVIEW",
                  "reason": "한국어로 아주 짧은 이유"
                }

                이제 아래 리뷰를 분석해.

                [REVIEW START]
                %s
                [REVIEW END]
                """.formatted(reviewText);
    }
}
