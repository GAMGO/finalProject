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
    private final ObjectMapper objectMapper;         // Spring Boot 기본 제공

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

        // 0️⃣ 공백이면 그냥 허용
        if (reviewText == null || reviewText.trim().isEmpty()) {
            ModerationResult r = new ModerationResult();
            r.setDecision("ALLOW");
            r.setReason("empty_or_whitespace");
            return r;
        }

        // 1️⃣ 우리 쪽 로컬 필터 먼저 적용 (욕설/인신공격 확실한 것들)
        if (containsLocalAbuse(reviewText)) {
            ModerationResult r = new ModerationResult();
            r.setDecision("BLOCK");
            r.setReason("local_abuse_match");
            log.info("[MOD] BLOCK by local rules: {}", reviewText);
            return r;
        }

        // 2️⃣ 나머지는 Gemini에게 판단 맡김
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
            // 🔁 에러 나면 "막는" 게 아니라 일단 통과 (UX 우선)
            log.error("Gemini moderation error, ALLOW for safety", e);
            ModerationResult r = new ModerationResult();
            r.setDecision("ALLOW");
            r.setReason("gemini_error_allow");
            return r;
        }
    }

    /* ======================  로컬 룰 ====================== */

    /**
     * 욕설/인신공격을 우리 쪽에서 직접 감지
     */
    private boolean containsLocalAbuse(String text) {
        String t = text;
        if (t == null) return false;

        // 공백 제거한 버전 (예: "못 생겼어요" -> "못생겼어요")
        String noSpace = t.replaceAll("\\s+", "");

        // 1) 대표적인 욕설 / 비속어
        String[] badWords = {
                "씨발", "시발", "ㅅㅂ", "씨팔", "ㅆㅂ",
                "개새끼", "개 새끼", "개새키", "개같은",
                "병신", "븅신", "ㅂㅅ",
                "좆", "존나", "꺼져",
                "장애인새끼", "장애인같은새끼"
        };

        for (String bw : badWords) {
            if (t.contains(bw) || noSpace.contains(bw.replaceAll("\\s+", ""))) {
                return true;
            }
        }

        // 2) 사장/주인 등 + 외모 비하 ("못생겼", "못 생겼")
        boolean hasTarget =
                t.contains("사장") ||
                t.contains("주인") ||
                t.contains("아저씨") ||
                t.contains("아줌마") ||
                t.contains("직원");

        boolean hasLookInsult =
                t.contains("못생겼") ||
                t.contains("못 생겼") ||
                noSpace.contains("못생긴") ||
                noSpace.contains("얼굴이별로") ||
                t.contains("얼굴이 별로") ||
                t.contains("추해") ||
                t.contains("징그러");

        if (hasTarget && hasLookInsult) {
            return true;
        }

        // 3) 그냥 외모 비하 문장 (대상을 명시적으로 안 써도)
        String[] lookInsults = {
                "못생겼다", "못 생겼다", "못생겼어요", "못 생겼어요",
                "얼굴이 더럽다", "얼굴이 별로다", "얼굴이 더러움",
                "추하게 생겼", "징그럽게 생겼"
        };
        for (String li : lookInsults) {
            if (t.contains(li) || noSpace.contains(li.replaceAll("\\s+", ""))) {
                return true;
            }
        }

        return false;
    }

    /* ======================  Gemini 응답 처리 ====================== */

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

    /* ======================  프롬프트 ====================== */

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
                  (특히 외모를 비하하는 표현: "사장님 못생겼어요" 등)
                - 고의적인 도배, 광고, 스팸(전화번호, 링크 등 홍보 목적)
                - 협박, 폭력 선동, 스토킹성 발언

                [ALLOW 기준]
                - 음식이 맛없다고 솔직하게 불평하는 리뷰
                  (예: "진짜 맛없음", "두 번은 안 시킬 듯")
                - 간이 세다/짜다/매웁다/양이 적다/비싸다/대기시간이 길다 같은
                  맛·양·가격·대기시간 관련 부정적 표현
                - 위생/청결/분위기/서비스가 별로였다는 평가
                  (예: "위생이 아쉬웠어요", "사장님이 좀 불친절했어요")
                - 욕설 없이 강한 표현, 감정 표현

                [REVIEW 기준]
                - 애매해서 자동으로 BLOCK하기 애매한 경우
                - 농담인지 모욕인지 판단이 애매한 표현

                [예시]
                입력: "야 이 장애인 같은 새끼야 가게 접어라"
                출력: {"decision":"BLOCK","reason":"장애인 비하와 심한 욕설"}

                입력: "사장님 못생겼어요"
                출력: {"decision":"BLOCK","reason":"사장님의 외모를 비하하는 인신 공격"}

                입력: "그 간이 좀 센 편이라 짠 맛이 강해요"
                출력: {"decision":"ALLOW","reason":"맛에 대한 부정적 평가"}

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
