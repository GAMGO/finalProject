package org.iclass.finalproject.user.controller;

import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.iclass.finalproject.common.ApiResponse;
import org.iclass.finalproject.common.CurrentUser;
import org.iclass.finalproject.user.dto.ChangePasswordRequest;
import org.iclass.finalproject.user.dto.LoginHistoryResponse;
import org.iclass.finalproject.user.dto.SetSecurityQuestionsRequest;
import org.iclass.finalproject.user.dto.UserProfileResponse;
import org.iclass.finalproject.user.service.LoginHistoryService;
import org.iclass.finalproject.user.service.UserService;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/me")
@RequiredArgsConstructor
public class MeController {

    private final UserService userService;
    private final LoginHistoryService loginHistoryService;

    // 내 프로필 조회
    @GetMapping
    public ApiResponse<UserProfileResponse> getProfile() {
        Long userId = CurrentUser.getUserId();
        return ApiResponse.ok(userService.getProfile(userId));
    }

    // 로그인 기록 + 총 일수 + 연속일수
    @GetMapping("/login-history")
    public ApiResponse<LoginHistoryResponse> getLoginHistory() {
        Long userId = CurrentUser.getUserId();
        LoginHistoryResponse res = loginHistoryService.getHistory(userId);
        return ApiResponse.ok(res);
    }

    // 현재 비밀번호로 비밀번호 변경
    @PostMapping("/change-password")
    public ApiResponse<Void> changePassword(@RequestBody @Valid ChangePasswordRequest req) {
        Long userId = CurrentUser.getUserId();
        userService.changePassword(userId, req);
        return ApiResponse.ok(null);
    }

    // 보안 질문/답변 설정
    @PostMapping("/security-questions")
    public ApiResponse<Void> setSecurityQuestions(@RequestBody @Valid SetSecurityQuestionsRequest req) {
        Long userId = CurrentUser.getUserId();
        userService.setSecurityQuestions(userId, req);
        return ApiResponse.ok(null);
    }
}

/*
 * [파일 설명]
 * - "내 정보" 관련 API(프로필, 로그인 기록, 비밀번호 변경, 보안질문 설정)를 제공하는 컨트롤러.
 * - /api/me/login-history는 날짜 리스트뿐만 아니라
 *   총 로그인 일수, 현재 연속 로그인 일수까지 함께 내려준다.
 */
