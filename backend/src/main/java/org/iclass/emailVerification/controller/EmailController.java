package org.iclass.emailVerification.controller;

import org.iclass.emailVerification.dto.EmailVerificationRequest;
import org.iclass.emailVerification.dto.EmailVerificationResponse;
import org.iclass.emailVerification.service.EmailService;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/email")
@RequiredArgsConstructor
@Slf4j
public class EmailController {

    private final EmailService emailService;

    @PostMapping("/verify")
    public ResponseEntity<EmailVerificationResponse> verifyEmail(@Valid @RequestBody EmailVerificationRequest request) {
        boolean isValid = emailService.verifyEmailCode(request.getEmail(), request.getToken(), "SIGNUP"); 

        if (!isValid) {
            log.warn("이메일 인증 실패: {}", request.getEmail());
            return ResponseEntity.badRequest()
                    .body(EmailVerificationResponse.failure("인증번호가 틀렸거나 만료되었습니다. 다시 시도해주세요."));
        }

        log.info("이메일 인증 성공: {}", request.getEmail());
        return ResponseEntity.ok(EmailVerificationResponse.success(request.getEmail()));
    }

    @PostMapping("/resend")
    public ResponseEntity<EmailVerificationResponse> resendVerificationEmail(@RequestParam String email) {
        emailService.sendCodeToEmail(email, "SIGNUP");
        return ResponseEntity.ok(EmailVerificationResponse.builder()
                .success(true)
                .message("인증 이메일이 발송되었습니다.")
                .email(email)
                .build());
    }
}