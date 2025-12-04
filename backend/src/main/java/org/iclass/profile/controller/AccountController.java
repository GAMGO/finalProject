// src/main/java/org/iclass/profile/controller/AccountController.java
package org.iclass.profile.controller;

import org.iclass.profile.dto.AccountInfoDto;
import org.iclass.profile.dto.AccountUpdateRequest;
import org.iclass.profile.dto.PasswordResetRequest;
import org.iclass.profile.dto.ProfileMessageResponse;
import org.iclass.profile.service.AccountService;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.web.bind.annotation.*;

import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;

@RequiredArgsConstructor
@RestController
@RequestMapping("/api/profile/account")
public class AccountController {

    private final AccountService accountService;

    // ----------------------------------------------------------
    // 1) 회원정보 조회
    //    GET /api/profile/account
    // ----------------------------------------------------------
    @GetMapping
    public AccountInfoDto getMyAccount(@AuthenticationPrincipal UserDetails user) {
        return accountService.getMyAccount(user);
    }

    // ----------------------------------------------------------
    // 2) 회원정보 수정
    //    PUT /api/profile/account
    // ----------------------------------------------------------
    @PutMapping
    public AccountInfoDto updateMyAccount(
            @AuthenticationPrincipal UserDetails user,
            @RequestBody @Valid AccountUpdateRequest request
    ) {
        return accountService.updateMyAccount(user, request);
    }

    // ----------------------------------------------------------
    // 3) 비밀번호 변경용 코드 이메일 발송
    //    POST /api/profile/account/password/code
    // ----------------------------------------------------------
    @PostMapping("/password/code")
    public ProfileMessageResponse sendPasswordResetCode(
            @AuthenticationPrincipal UserDetails user
    ) {
        accountService.sendPasswordResetCode(user);
        return new ProfileMessageResponse("비밀번호 변경용 인증 코드를 이메일로 전송했습니다.");
    }

    // ----------------------------------------------------------
    // 4) 코드 검증 + 비밀번호 변경
    //    POST /api/profile/account/password/reset
    // ----------------------------------------------------------
    @PostMapping("/password/reset")
    public ProfileMessageResponse resetPassword(
            @AuthenticationPrincipal UserDetails user,
            @RequestBody @Valid PasswordResetRequest request
    ) {
        accountService.resetPassword(user, request);
        return new ProfileMessageResponse("비밀번호가 변경되었습니다.");
    }
}
