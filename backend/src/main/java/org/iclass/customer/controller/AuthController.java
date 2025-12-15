package org.iclass.customer.controller;

import java.time.LocalDateTime;
import java.util.Map;
import java.util.Optional;

import org.iclass.BalcklistedToken.service.TokenBlacklistService;
import org.iclass.customer.dto.LoginRequest;
import org.iclass.customer.dto.LoginResponse;
import org.iclass.customer.dto.LogoutResponse;
import org.iclass.customer.dto.SignupRequest;
import org.iclass.customer.dto.SignupResponse;
import org.iclass.customer.dto.TokenRefreshRequest;
import org.iclass.customer.dto.TokenRefreshResponse;
import org.iclass.customer.entity.CustomersEntity;
import org.iclass.customer.repository.CustomersRepository;
import org.iclass.customer.service.CustomersService;
import org.iclass.security.JwtTokenProvider;

import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseCookie;
import org.springframework.http.ResponseEntity;

import org.springframework.security.authentication.AuthenticationManager;
import org.springframework.security.authentication.BadCredentialsException;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;

import org.springframework.security.core.Authentication;
import org.springframework.security.core.GrantedAuthority;
import org.springframework.security.core.annotation.AuthenticationPrincipal;

import org.springframework.security.core.userdetails.UserDetails;

import org.springframework.util.StringUtils;
import org.springframework.web.bind.annotation.*;

import io.swagger.v3.oas.annotations.Parameter;
import io.swagger.v3.oas.annotations.enums.ParameterIn;

import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import jakarta.validation.Valid;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;

@Slf4j
@RequiredArgsConstructor
@RestController
@RequestMapping("/api/auth")
public class AuthController {

    private final CustomersService customersService;
    private final AuthenticationManager authenticationManager;
    private final JwtTokenProvider jwtTokenProvider;
    private final TokenBlacklistService tokenBlacklistService;
    private final CustomersRepository customersRepository;

    // CustomersEntity를 그대로 반환해서 비밀번호 같은 민감한 정보가 노출될 수 있다고 함.
    // 자동 로그인을 위해 회원가입 엔드포인트에서 토큰 처리
    @PostMapping("/signup")
    public ResponseEntity<SignupResponse> signup(
            @Valid @RequestBody SignupRequest request,
            HttpServletResponse response) {

        CustomersEntity saved = customersService.signup(request);

        Authentication authentication = authenticationManager.authenticate(
                new UsernamePasswordAuthenticationToken(request.getId(), request.getPassword()));

        String token = jwtTokenProvider.createToken(authentication);
        String refreshToken = jwtTokenProvider.createRefreshToken(authentication);

        saved.setRefreshToken(refreshToken);
        customersRepository.save(saved);

        ResponseCookie cookie = ResponseCookie.from("refreshToken", refreshToken)
                .httpOnly(true)
                .secure(true) // 로컬이면 false로 바꿀 수 있음
                .path("/")
                .maxAge(7 * 24 * 60 * 60)
                .sameSite("Lax")
                .build();

        response.addHeader(HttpHeaders.SET_COOKIE, cookie.toString());

        SignupResponse res = SignupResponse.fromEntity(saved, token);
        return ResponseEntity.ok(res);
    }

    @PostMapping("/login")
    public ResponseEntity<?> login(@Valid @RequestBody LoginRequest request) {
        try {
            Authentication authentication = authenticationManager.authenticate(
                    new UsernamePasswordAuthenticationToken(request.getId(), request.getPassword()));

            String token = jwtTokenProvider.createToken(authentication);
            String refreshToken = jwtTokenProvider.createRefreshToken(authentication);

            // ✅ role 계산 (ROLE_ADMIN이면 ADMIN)
            String role = authentication.getAuthorities().stream()
                    .map(GrantedAuthority::getAuthority)
                    .anyMatch("ROLE_ADMIN"::equals)
                    ? "ADMIN" : "USER";

            // ✅ (기존) Refresh Token DB 저장: admin은 DB 유저가 없으니 그냥 스킵됨
            String userId = request.getId();
            Optional<Long> idxOpt = customersRepository.findIdxByUsername(userId);
            if (idxOpt.isPresent()) {
                Long idx = idxOpt.get();
                Optional<CustomersEntity> userEntityOpt = customersRepository.findByIdx(idx);
                if (userEntityOpt.isPresent()) {
                    CustomersEntity user = userEntityOpt.get();
                    user.setRefreshToken(refreshToken);
                    customersRepository.save(user);
                    log.info("User {}'s Refresh Token saved. (idx: {})", userId, idx);
                } else {
                    log.warn("Login OK but user entity not found for idx: {}", idx);
                }
            } else {
                log.warn("Login OK but no idx for username: {}", userId);
            }

            UserDetails userDetails = (UserDetails) authentication.getPrincipal();
            LoginResponse response = LoginResponse.builder()
                    .token(token)
                    .refreshToken(refreshToken)
                    .tokenType("Bearer")
                    .id(userDetails.getUsername())
                    .role(role) // ✅ 추가
                    .build();

            return ResponseEntity.ok(response);

        } catch (BadCredentialsException e) {
            return ResponseEntity.status(401).body(
                    Map.of("error", "invalid_credentials",
                           "message", "아이디 또는 비밀번호가 올바르지 않습니다."));
        }
    }

    @PostMapping("/logout")
    public ResponseEntity<LogoutResponse> logout(
            @Parameter(in = ParameterIn.HEADER, name = HttpHeaders.AUTHORIZATION,
                    description = "Bearer <JWT>", required = false)
            @RequestHeader(value = HttpHeaders.AUTHORIZATION, required = false) String authorization,
            @AuthenticationPrincipal UserDetails user,
            HttpServletRequest request) {

        if (!StringUtils.hasText(authorization)) {
            authorization = request.getHeader(HttpHeaders.AUTHORIZATION);
        }

        String token = null;
        if (StringUtils.hasText(authorization)) {
            token = authorization.startsWith("Bearer ")
                    ? authorization.substring(7)
                    : authorization.trim();
        }

        if (!StringUtils.hasText(token)) {
            return ResponseEntity.badRequest()
                    .body(LogoutResponse.builder()
                            .message("Missing Authorization header (expected: Bearer <token>)")
                            .build());
        }

        String id = (user != null) ? user.getUsername() : jwtTokenProvider.getUsernameFromToken(token);
        LocalDateTime exp = jwtTokenProvider.getExpiry(token);

        if (!StringUtils.hasText(id) || exp == null) {
            return ResponseEntity.badRequest()
                    .body(LogoutResponse.builder().message("Invalid token").build());
        }

        try {
            Optional<Long> idxOpt = customersRepository.findIdxByUsername(id);
            if (idxOpt.isPresent()) {
                Long idx = idxOpt.get();
                Optional<CustomersEntity> userEntityOpt = customersRepository.findByIdx(idx);
                if (userEntityOpt.isPresent()) {
                    CustomersEntity customersEntity = userEntityOpt.get();
                    customersEntity.setRefreshToken(null);
                    customersRepository.save(customersEntity);
                    log.info("Refresh Token invalidated (id: {}, idx: {})", id, idx);
                } else {
                    log.warn("Logout: CustomersEntity not found by idx({})", idx);
                }
            } else {
                log.warn("Logout: idx not found for id({})", id);
            }
        } catch (Exception e) {
            log.error("Failed to invalidate refresh token for user {}: {}", id, e.getMessage());
        }

        tokenBlacklistService.blacklist(token, id, exp, "USER_LOGOUT");
        return ResponseEntity.ok(LogoutResponse.builder().message("Logged out").build());
    }

    @PostMapping("/refresh")
    public ResponseEntity<?> refreshToken(@RequestBody TokenRefreshRequest request) {
        String refreshToken = request.getRefreshToken();

        if (!StringUtils.hasText(refreshToken) || !jwtTokenProvider.validateToken(refreshToken)) {
            return ResponseEntity.status(HttpStatus.FORBIDDEN)
                    .body(Map.of("error", "Invalid or expired refresh token. Please log in again."));
        }

        Optional<CustomersEntity> userOpt = customersRepository.findByRefreshToken(refreshToken);
        if (userOpt.isEmpty()) {
            log.warn("Invalid refresh token detected: {}", refreshToken);
            return ResponseEntity.status(HttpStatus.FORBIDDEN)
                    .body(Map.of("error", "Refresh token mismatch or user not found."));
        }

        Authentication authentication = jwtTokenProvider.getAuthentication(refreshToken);
        String token = jwtTokenProvider.createToken(authentication);

        return ResponseEntity.ok(
                TokenRefreshResponse.builder()
                        .token(token)
                        .refreshToken(refreshToken)
                        .tokenType("Bearer")
                        .build());
    }

    @GetMapping("/check-id")
    public ResponseEntity<?> checkId(@RequestParam String id) {
        boolean isAvailable = !customersRepository.existsById(id);
        return ResponseEntity.ok(Map.of("available", isAvailable));
    }

    @GetMapping("/check-email")
    public ResponseEntity<?> checkEmail(@RequestParam String email) {
        boolean isAvailable = customersRepository.findByEmail(email).isEmpty();
        return ResponseEntity.ok(Map.of("available", isAvailable));
    }
}
