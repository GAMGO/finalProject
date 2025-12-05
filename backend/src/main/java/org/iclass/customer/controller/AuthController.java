package org.iclass.customer.controller;

import java.net.URI;
import java.time.LocalDateTime;
import java.util.Map;
import java.util.Optional;

import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.security.authentication.AuthenticationManager;
import org.springframework.security.authentication.BadCredentialsException;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestHeader;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.util.StringUtils;
import org.iclass.customer.dto.LoginRequest;
import org.iclass.customer.dto.LoginResponse;
import org.iclass.customer.dto.LogoutResponse;
import org.iclass.customer.dto.SignupRequest;
import org.iclass.customer.dto.SignupResponse;
import org.iclass.customer.dto.TokenRefreshRequest;
import org.iclass.customer.dto.TokenRefreshResponse;
import org.iclass.customer.entity.CustomersEntity;
import org.iclass.security.JwtTokenProvider;
import org.iclass.customer.service.CustomersService;
import org.iclass.BalcklistedToken.service.TokenBlacklistService;
import org.iclass.customer.repository.CustomersRepository;
import io.swagger.v3.oas.annotations.Parameter;
import io.swagger.v3.oas.annotations.enums.ParameterIn;
import jakarta.servlet.http.HttpServletRequest;
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

    // 지금까지는 CustomersEntity를 그대로 반환해서 비밀번호 같은 민감한 정보가 노출됐음
    // 응답 전용 DTO(SignupResponse)로 변환해서 필요한 데이터만 반환
    @PostMapping("/signup")
    public ResponseEntity<SignupResponse> signup(@Valid @RequestBody SignupRequest request) {
        CustomersEntity saved = customersService.signup(request);
        SignupResponse response = SignupResponse.fromEntity(saved);
        return ResponseEntity.created(URI.create("/api/users/" + saved.getId()))
                .body(response);
    }

    @PostMapping("/login")
    public ResponseEntity<?> login(@Valid @RequestBody LoginRequest request) { // >>> [CHANGED] 타입만 와일드카드
        try {
            // 사용자 인증
            Authentication authentication = authenticationManager.authenticate(
                    new UsernamePasswordAuthenticationToken(
                            request.getId(),
                            request.getPassword()));

            // JWT 토큰 생성
            String token = jwtTokenProvider.createToken(authentication);
            String refreshToken = jwtTokenProvider.createRefreshToken(authentication);
            String userId = request.getId();
            Optional<CustomersEntity> userEntityOpt = Optional.empty();
            Optional<Long> idxOpt = customersRepository.findIdxByUsername(userId);
            if (idxOpt.isPresent()) {
                Long idx = idxOpt.get();
                userEntityOpt = customersRepository.findByIdx(idx);
            }
            if (userEntityOpt.isPresent()) {
                CustomersEntity user = userEntityOpt.get();
                // Refresh Token 값을 엔티티에 설정
                user.setRefreshToken(refreshToken);
                // DB에 변경 사항 저장 (영속화)
                customersRepository.save(user);
                log.info("User {}'s Refresh Token successfully saved to DB. (idx: {})", userId, user.getIdx());
            } else {
                // 사용자 인증은 성공했으나 DB에서 엔티티를 찾지 못한 경우
                log.warn("Login successful but failed to find user for ID: {}", userId);
            }
            // 사용자 정보 조회 -> Principal만 가져오기
            UserDetails userDetails = (UserDetails) authentication.getPrincipal();
            LoginResponse response = LoginResponse.builder()
                    .token(token)
                    .refreshToken(refreshToken)
                    .tokenType("Bearer")
                    .id(userDetails.getUsername())
                    .build();

            return ResponseEntity.ok(response);

        } catch (BadCredentialsException e) {
            // 로그인 실패 시 401 + 명확한 메시지(JSON) 반환
            return ResponseEntity.status(401).body(
                    Map.of(
                            "error", "invalid_credentials",
                            "message", "아이디 또는 비밀번호가 올바르지 않습니다."));
        }
    }

    @PostMapping("/logout")
    public ResponseEntity<LogoutResponse> logout(
            @Parameter(in = ParameterIn.HEADER, name = HttpHeaders.AUTHORIZATION, description = "Bearer <JWT>", required = false) @RequestHeader(value = HttpHeaders.AUTHORIZATION, required = false) String authorization,
            @AuthenticationPrincipal UserDetails user,
            HttpServletRequest request) {

        // 전역 Authorize가 안 붙거나 프록시에서 빠질 수 있어 추가 확인
        if (!StringUtils.hasText(authorization)) {
            authorization = request.getHeader(HttpHeaders.AUTHORIZATION);
        }

        // Bearer 접두사 유무 모두 허용
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

        // 토큰에서 사용자/만료시각 추출 (메서드명은 현재 구현과 동일 사용)
        String id = (user != null) ? user.getUsername() : jwtTokenProvider.getUsernameFromToken(token);
        LocalDateTime exp = jwtTokenProvider.getExpiry(token);

        if (!StringUtils.hasText(id) || exp == null) {
            return ResponseEntity.badRequest()
                    .body(LogoutResponse.builder().message("Invalid token").build());
        }
        try {
            // 1. String ID(username)로 Long 타입의 idx를 조회
            Optional<Long> idxOpt = customersRepository.findIdxByUsername(id);

            if (idxOpt.isPresent()) {
                Long idx = idxOpt.get();
                // 2. Long idx로 CustomersEntity 조회 (findByIdx 사용)
                Optional<CustomersEntity> userEntityOpt = customersRepository.findByIdx(idx);

                if (userEntityOpt.isPresent()) {
                    CustomersEntity customersEntity = userEntityOpt.get();
                    customersEntity.setRefreshToken(null);
                    customersRepository.save(customersEntity);
                    log.info("User {}'s Refresh Token invalidated in DB. (idx: {})", id, idx);
                } else {
                    log.warn("Refresh Token 무효화 실패: idx({})로 CustomersEntity를 찾을 수 없습니다.", idx);
                }
            } else {
                log.warn("Refresh Token 무효화 실패: 사용자 ID({})에 해당하는 idx를 찾을 수 없습니다.", id);
            }
        } catch (Exception e) {
            // DB 또는 트랜잭션 관련 예외가 발생했을 경우
            log.error("Failed to invalidate refresh token for user {}: {}", id, e.getMessage());
            // 🚨 이 예외로 인해 500 에러가 발생했을 수 있습니다.
            // 만약 이 예외가 계속 발생한다면, 이 로직을 CustomersService의 `@Transactional` 메서드 안으로 옮기는 것을
            // 고려해야 합니다.
        }
        // Access Token 블랙리스트 처리 (DB 문제와 관계없이 진행)
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

        // Refresh Token으로 사용자 찾기 (DB에 저장된 토큰인지 확인)
        Optional<CustomersEntity> userOpt = customersRepository.findByRefreshToken(refreshToken);
        if (userOpt.isEmpty()) {
            // 토큰 불일치 (탈취 또는 이미 로그아웃된 토큰)
            log.warn("Invalid refresh token detected: {}", refreshToken);
            return ResponseEntity.status(HttpStatus.FORBIDDEN)
                    .body(Map.of("error", "Refresh token mismatch or user not found."));
        }
        CustomersEntity user = userOpt.get();
        Authentication authentication = jwtTokenProvider.getAuthentication(refreshToken);
        // 2. 새로운 Access Token 발급
        String token = jwtTokenProvider.createToken(authentication);
        // 3. 응답 반환
        return ResponseEntity.ok(
                TokenRefreshResponse.builder()
                        .token(token)
                        .refreshToken(refreshToken) // Refresh Token은 재사용 (선택적으로 새로운 토큰을 발급하고 DB에 업데이트 가능)
                        .tokenType("Bearer")
                        .build());
    }
}