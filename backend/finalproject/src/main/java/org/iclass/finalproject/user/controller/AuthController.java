package org.iclass.finalproject.user.controller;

import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpSession;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.iclass.finalproject.common.ApiResponse;
import org.iclass.finalproject.common.CurrentUser;
import org.iclass.finalproject.user.dto.LoginRequest;
import org.iclass.finalproject.user.dto.ResetPasswordByQuestionRequest;
import org.iclass.finalproject.user.dto.SignupRequest;
import org.iclass.finalproject.user.model.User;
import org.iclass.finalproject.user.service.LoginHistoryService;
import org.iclass.finalproject.user.service.UserService;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/auth")
@RequiredArgsConstructor
public class AuthController {

    private final UserService userService;
    private final LoginHistoryService loginHistoryService;

    @PostMapping("/signup")
    public ApiResponse<Long> signup(@RequestBody @Valid SignupRequest req) {
        User u = userService.signup(req);
        return ApiResponse.ok(u.getId());
    }

    @PostMapping("/login")
    public ApiResponse<String> login(@RequestBody @Valid LoginRequest req,
                                     HttpServletRequest request) {
        // 이메일/비밀번호 검증
        User user = userService.login(req);

        // 세션에 userId 저장
        HttpSession session = request.getSession(true);
        session.setAttribute(CurrentUser.SESSION_KEY, user.getId());

        // 로그인 기록 저장
        loginHistoryService.recordLogin(user.getId());

        // 프론트는 세션 쿠키(JSESSIONID)로 로그인 상태 유지
        return ApiResponse.ok("LOGIN_SUCCESS");
    }

    @PostMapping("/logout")
    public ApiResponse<Void> logout(HttpServletRequest request) {
        HttpSession session = request.getSession(false);
        if (session != null) {
            session.invalidate();
        }
        return ApiResponse.ok(null);
    }

    // 비밀번호 찾기: 이메일 + 보안질문으로 재설정
    @PostMapping("/reset-password/question")
    public ApiResponse<Void> resetPasswordByQuestion(@RequestBody @Valid ResetPasswordByQuestionRequest req) {
        userService.resetPasswordByQuestion(req);
        return ApiResponse.ok(null);
    }
}

/*
 * [파일 설명]
 * - 회원가입/로그인/로그아웃/비밀번호 재설정(보안질문 기반) API 엔드포인트.
 * - /login:
 *   1) 이메일+비밀번호 검증 (UserService.login)
 *   2) 세션에 USER_ID 저장
 *   3) 로그인 히스토리 기록
 * - /logout:
 *   세션 삭제로 로그아웃 처리.
 * - /reset-password/question:
 *   이메일+보안질문/답으로 비밀번호 재설정.
 */
