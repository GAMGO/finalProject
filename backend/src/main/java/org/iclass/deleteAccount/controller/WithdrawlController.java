// WithdrawlController.java
package org.iclass.deleteAccount.controller;

import lombok.RequiredArgsConstructor;
import org.iclass.customer.service.CustomersService;
import org.springframework.http.ResponseEntity;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.web.bind.annotation.*;
import java.util.Map;

@RestController
@RequestMapping("/api/auth")
@RequiredArgsConstructor
public class WithdrawlController {

    private final CustomersService customersService;

    @DeleteMapping("/withdrawal")
    public ResponseEntity<?> withdraw(@AuthenticationPrincipal UserDetails userDetails) {
        if (userDetails == null) {
            return ResponseEntity.status(401).body(Map.of("message", "인증 정보가 없습니다."));
        }

        try {
            // 서비스의 소프트 삭제 로직 호출
            customersService.deleteCustomer(userDetails.getUsername());
            return ResponseEntity.ok(Map.of("message", "회원 탈퇴가 완료되었습니다."));
        } catch (Exception e) {
            return ResponseEntity.status(500).body(Map.of("message", "탈퇴 처리 중 오류 발생"));
        }
    }
}