package org.iclass.finalproject.owner.controller;

import lombok.Data;
import lombok.RequiredArgsConstructor;
import org.iclass.finalproject.common.ApiResponse;
import org.iclass.finalproject.common.CurrentUser;
import org.iclass.finalproject.owner.model.OwnerVerification;
import org.iclass.finalproject.owner.service.OwnerVerificationService;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequiredArgsConstructor
public class OwnerVerificationController {

    private final OwnerVerificationService ownerVerificationService;

    @Data
    public static class OwnerRequestBody {
        private String memo;
    }

    @PostMapping("/api/places/{placeId}/owner-verification")
    public ApiResponse<Long> requestOwner(@PathVariable Long placeId,
                                          @RequestBody(required = false) OwnerRequestBody body) {
        Long userId = CurrentUser.getUserId();
        String memo = body != null ? body.getMemo() : null;
        OwnerVerification ov = ownerVerificationService.request(userId, placeId, memo);
        return ApiResponse.ok(ov.getId());
    }

    // 관리자용 예시
    @GetMapping("/api/admin/owner-verification")
    public ApiResponse<List<OwnerVerification>> listPending() {
        return ApiResponse.ok(ownerVerificationService.listPending());
    }

    @PostMapping("/api/admin/owner-verification/{id}/approve")
    public ApiResponse<Void> approve(@PathVariable Long id) {
        ownerVerificationService.approve(id);
        return ApiResponse.ok(null);
    }

    @PostMapping("/api/admin/owner-verification/{id}/reject")
    public ApiResponse<Void> reject(@PathVariable Long id,
                                    @RequestBody(required = false) OwnerRequestBody body) {
        String memo = body != null ? body.getMemo() : null;
        ownerVerificationService.reject(id, memo);
        return ApiResponse.ok(null);
    }
}

/*
 * [파일 설명]
 * - 사장 인증 요청/승인/거절 REST 엔드포인트.
 * - 사장은 /api/places/{id}/owner-verification 로 요청을 보내고,
 *   관리자는 /api/admin/... 엔드포인트로 검토/처리한다는 흐름을 가정.
 */
