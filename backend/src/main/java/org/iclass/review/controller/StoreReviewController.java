// src/main/java/org/iclass/review/controller/StoreReviewController.java
package org.iclass.review.controller;

import jakarta.validation.Valid;
import org.iclass.common.ApiResponse;
import org.iclass.customer.repository.CustomersRepository;
import org.iclass.review.dto.*;
import org.iclass.review.service.StoreReviewService;
import org.springframework.data.domain.*;
import org.springframework.http.ResponseEntity;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.web.bind.annotation.*;

// 🔥 네가 만든 모더레이션 서비스 / DTO (패키지명은 실제에 맞게 수정)
import org.iclass.gemini.ReviewModerationService;
import org.iclass.gemini.dto.ModerationResult;

@RestController
@RequestMapping("/api/stores/{storeIdx}/reviews")
public class StoreReviewController {

    private final StoreReviewService service;
    private final CustomersRepository customersRepository;
    private final ReviewModerationService reviewModerationService;   // 🔥 추가

    public StoreReviewController(StoreReviewService service,
                                 CustomersRepository customersRepository,
                                 ReviewModerationService reviewModerationService) { // 🔥 추가
        this.service = service;
        this.customersRepository = customersRepository;
        this.reviewModerationService = reviewModerationService;
    }

    // ✅ CustomUserDetails 없이: SecurityContext의 username -> DB에서 idx 조회
    private Long currentUserId() {
        Authentication auth = SecurityContextHolder.getContext().getAuthentication();
        if (auth == null || auth.getPrincipal() == null) {
            throw new IllegalStateException("로그인 사용자를 찾을 수 없습니다.");
        }

        String username;
        Object principal = auth.getPrincipal();

        if (principal instanceof UserDetails ud) {
            username = ud.getUsername();
        } else if (principal instanceof String s) {
            username = s; // 일부 환경에선 principal이 문자열(username)로 들어옵니다.
        } else {
            throw new IllegalStateException("지원하지 않는 인증 주체 타입: " + principal.getClass().getName());
        }

        return customersRepository.findIdxByUsername(username)
                .orElseThrow(() -> new IllegalStateException("사용자 정보를 찾을 수 없습니다: " + username));
    }

    // (선택) 관리자 권한 체크 - ROLE_ADMIN 유무
    private boolean isAdmin() {
        Authentication auth = SecurityContextHolder.getContext().getAuthentication();
        if (auth == null) return false;
        return auth.getAuthorities().stream()
                .anyMatch(a -> "ROLE_ADMIN".equals(a.getAuthority()));
    }

    // 🔥 리뷰 작성: 제미나이로 욕설/비하 필터 후 저장
    @PostMapping
    public ResponseEntity<?> create(@PathVariable Long storeIdx,
                                    @Valid @RequestBody StoreReviewRequest req) {

        // ⚠️ StoreReviewRequest 안에 필드 이름이 reviewText라고 가정
        // 만약 text / content 이런 이름이면 여기만 맞게 바꿔줘
        String text = req.getReviewText();

        // 1️⃣ 모더레이션 호출
        ModerationResult moderation = reviewModerationService.moderate(text);

        if (moderation != null && moderation.isBlocked()) {
            // BLOCK이면 바로 400 리턴 (프론트에서는 status 코드만 보고 alert 띄우고 있음)
            return ResponseEntity
                    .badRequest()
                    .body("욕설·비하·스팸 등으로 판단되어 등록할 수 없는 리뷰입니다.");
        }

        // (원하면 REVIEW 상태도 따로 처리 가능)
        // if (moderation != null && moderation.needManualReview()) { ... }

        // 2️⃣ 통과한 경우 정상 저장
        Long id = service.create(storeIdx, currentUserId(), req);
        return ResponseEntity.ok(id);   // 기존처럼 ID 그대로 리턴 (프론트 로직 안 깨짐)
    }

    // 🔥 리뷰 수정에도 같은 필터 적용
    @PutMapping("/{reviewId}")
    public ResponseEntity<?> update(@PathVariable Long storeIdx,
                                    @PathVariable Long reviewId,
                                    @Valid @RequestBody StoreReviewRequest req) {

        String text = req.getReviewText();
        ModerationResult moderation = reviewModerationService.moderate(text);

        if (moderation != null && moderation.isBlocked()) {
            return ResponseEntity
                    .badRequest()
                    .body("욕설·비하·스팸 등으로 판단되어 수정할 수 없는 리뷰입니다.");
        }

        service.update(reviewId, currentUserId(), req, isAdmin());
        return ResponseEntity.noContent().build();   // 기존 로직 유지
    }

    // 기존: Page 자체 내려주는 목록
    @GetMapping
    public ResponseEntity<Page<StoreReviewResponse>> list(@PathVariable Long storeIdx,
                                                          @RequestParam(defaultValue = "0") int page,
                                                          @RequestParam(defaultValue = "10") int size) {
        Page<StoreReviewResponse> result =
                service.list(storeIdx, PageRequest.of(page, size));
        return ResponseEntity.ok(result);
    }

    // 기존: 통계만
    @GetMapping("/stats")
    public ResponseEntity<StoreReviewStatsResponse> stats(@PathVariable Long storeIdx) {
        return ResponseEntity.ok(service.stats(storeIdx));
    }

    // 🔥 신규: 리뷰 + 통계 한 방에 (프론트에서 이거 씀)
    @GetMapping("/with-stats")
    public ResponseEntity<ApiResponse<StoreReviewListWithStatsResponse>> listWithStats(
            @PathVariable Long storeIdx,
            @RequestParam(defaultValue = "0") int page,
            @RequestParam(defaultValue = "10") int size) {

        StoreReviewListWithStatsResponse body =
                service.listWithStats(storeIdx, PageRequest.of(page, size));

        return ResponseEntity.ok(ApiResponse.success(body));
    }

    @DeleteMapping("/{reviewId}")
    public ResponseEntity<Void> delete(@PathVariable Long storeIdx,
                                       @PathVariable Long reviewId) {
        service.delete(reviewId, currentUserId(), isAdmin());
        return ResponseEntity.noContent().build();
    }
}
