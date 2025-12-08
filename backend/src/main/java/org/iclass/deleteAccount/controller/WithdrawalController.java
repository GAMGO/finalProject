package org.iclass.deleteAccount.controller;

import lombok.RequiredArgsConstructor;
import org.iclass.customer.service.CustomersService;
import org.iclass.deleteAccount.dto.RestoreRequest;
import org.iclass.security.JwtTokenProvider;
import org.iclass.BalcklistedToken.service.TokenBlacklistService; // 블랙리스트 서비스
import org.springframework.http.HttpHeaders;
import org.springframework.http.ResponseEntity;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.web.bind.annotation.*;
import org.springframework.util.StringUtils;
import java.time.LocalDateTime;
import java.util.Map;

@RestController
@RequestMapping("/api/auth")
@RequiredArgsConstructor
public class WithdrawalController {

    private final CustomersService customersService;
    private final TokenBlacklistService tokenBlacklistService;
    private final JwtTokenProvider jwtTokenProvider;

    /**
     * 회원 탈퇴 (소프트 삭제 및 블랙리스트 등록)
     */
    @DeleteMapping("/withdrawal")
    public ResponseEntity<?> withdraw(
            @RequestHeader(value = HttpHeaders.AUTHORIZATION, required = false) String authorization,
            @AuthenticationPrincipal UserDetails userDetails) {

        if (userDetails == null || !StringUtils.hasText(authorization)) {
            return ResponseEntity.status(401).body(Map.of("message", "인증 정보가 유효하지 않습니다."));
        }

        // 1. Access Token 블랙리스트 등록 (즉시 무효화)
        String token = authorization.startsWith("Bearer ") ? authorization.substring(7) : authorization;
        String id = userDetails.getUsername();
        LocalDateTime exp = jwtTokenProvider.getExpiry(token);
        tokenBlacklistService.blacklist(token, id, exp, "USER_WITHDRAWAL");

        // 2. 서비스 호출 (DB 상태 변경 및 이메일 발송)
        try {
            customersService.deleteCustomer(id);
            return ResponseEntity.ok(Map.of("message", "회원 탈퇴 처리되었습니다. 복구 코드가 이메일로 발송되었습니다."));
        } catch (Exception e) {
            return ResponseEntity.status(500).body(Map.of("message", "탈퇴 처리 중 오류 발생"));
        }
    }

    /**
     * 계정 복구 (이메일로 받은 토큰 사용)
     */
    @PostMapping("/restore")
    public ResponseEntity<?> restore(@RequestBody RestoreRequest request) {

        // request.get("recoveryToken") 대신 getter 사용
        String recoveryToken = request.getRecoveryToken();

        if (!StringUtils.hasText(recoveryToken)) {
            return ResponseEntity.badRequest().body(Map.of("message", "복구 코드가 필요합니다."));
        }

        try {
            customersService.restoreCustomer(recoveryToken);
            return ResponseEntity.ok(Map.of("message", "계정이 성공적으로 복구되었습니다. 다시 로그인해 주세요."));
        } catch (IllegalArgumentException e) {
            return ResponseEntity.status(400).body(Map.of("message", e.getMessage()));
        } catch (Exception e) {
            return ResponseEntity.status(500).body(Map.of("message", "복구 처리 중 서버 오류 발생"));
        }
    }
}