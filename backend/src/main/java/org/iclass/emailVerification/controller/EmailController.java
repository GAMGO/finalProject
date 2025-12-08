package org.iclass.emailVerification.controller;

import org.iclass.emailVerification.dto.EmailVerificationRequest;
import org.iclass.emailVerification.dto.EmailVerificationResponse;
import org.iclass.customer.entity.CustomersEntity;
import org.iclass.customer.repository.CustomersRepository;
import org.iclass.emailVerification.service.EmailService;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.time.LocalDateTime;
import java.util.Map;
import java.util.Optional;

@RestController
@RequestMapping("/api/email")
@RequiredArgsConstructor
@Slf4j
public class EmailController {

    private final EmailService emailService;
    private final CustomersRepository customersRepository;

    /**
     * 이메일 인증 확인
     * 
     * @param request 인증 요청 (토큰 포함)
     * @return 인증 결과
     */

    // 이메일 인증 확인
    // EmailController.java 내 verifyEmail 메서드 수정

@PostMapping("/verify")
public ResponseEntity<EmailVerificationResponse> verifyEmail(@Valid @RequestBody EmailVerificationRequest request) {
    try {
        String email = request.getEmail(); // ✅ DTO에서 이메일 추출
        String token = request.getToken();
        log.info("이메일 인증 요청: {} - {}", email, token);

        if (token == null || !token.matches("\\d{6}")) {
            return ResponseEntity.badRequest()
                    .body(EmailVerificationResponse.failure("6자리 인증번호를 입력해주세요"));
        }

        boolean isValid = emailService.verifyEmailCode(email, token, "SIGNUP"); 

        if (!isValid) {
            return ResponseEntity.badRequest()
                    .body(EmailVerificationResponse.failure("인증번호가 틀렸거나 만료되었습니다"));
        }

        // 3. 인증 성공 처리 (DB 업데이트 로직 제거 - 가입 시점에 처리됨)
        log.info("이메일 인증 성공: {}", email);

        // 프론트엔드가 이후에 /signup(POST)을 호출할 수 있도록 성공 응답 반환
        return ResponseEntity.ok(EmailVerificationResponse.success(email));

    } catch (Exception e) {
        log.error("이메일 인증 처리 중 오류 발생", e);
        return ResponseEntity.internalServerError()
                .body(EmailVerificationResponse.failure("서버 오류가 발생했습니다"));
    }
}

    /**
     * 이메일 인증 재발송
     * 
     * @param email 재발송할 이메일
     * @return 재발송 결과
     */
    @PostMapping("/resend")
    public ResponseEntity<EmailVerificationResponse> resendVerificationEmail(@RequestParam String email) {
        try {
            log.info("이메일 인증 재발송 요청: {}", email);

            // ✅ EmailService에 구현된 캐시 기반 발송 메서드 사용
            emailService.sendCodeToEmail(email, "SIGNUP"); // EmailService.java의 sendCodeToEmail 호출

            log.info("이메일 인증 재발송 완료: {}", email);

            return ResponseEntity.ok(EmailVerificationResponse.builder()
                    .success(true)
                    .message("인증 이메일이 재발송되었습니다")
                    .email(email)
                    .build());

        } catch (Exception e) {
            log.error("이메일 재발송 오류", e);
            return ResponseEntity.internalServerError()
                    .body(EmailVerificationResponse.failure("서버 오류가 발생했습니다"));
        }
    }

    /**
     * 이메일 인증 상태 확인
     * 
     * @param email 확인할 이메일
     * @return 인증 상태
     */

    // 이메일 인증 상태 확인
    @GetMapping("/status")
    public ResponseEntity<EmailVerificationResponse> getVerificationStatus(@RequestParam String email) {
        try {
            Optional<CustomersEntity> customerOpt = customersRepository.findByEmail(email);
            if (customerOpt.isEmpty()) {
                return ResponseEntity.badRequest()
                        .body(EmailVerificationResponse.failure("존재하지 않는 이메일입니다"));
            }

            CustomersEntity customer = customerOpt.get();
            boolean isVerified = customer.getEmailVerified();

            return ResponseEntity.ok(EmailVerificationResponse.builder()
                    .success(isVerified)
                    .message(isVerified ? "인증된 이메일입니다" : "인증되지 않은 이메일입니다")
                    .email(email)
                    .build());

        } catch (Exception e) {
            log.error("이메일 인증 상태 확인 중 오류 발생", e);
            return ResponseEntity.internalServerError()
                    .body(EmailVerificationResponse.failure("서버 오류가 발생했습니다"));
        }
    }

}