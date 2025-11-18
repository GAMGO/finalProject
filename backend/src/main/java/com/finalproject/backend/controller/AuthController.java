package com.finalproject.backend.controller;

import java.util.Map;

import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import com.finalproject.backend.config.JwtProvider;
import com.finalproject.backend.dto.SignupRequest;
import com.finalproject.backend.entity.Customers;
import com.finalproject.backend.service.AuthService;
import com.finalproject.backend.service.GoogleAuthService;

import lombok.RequiredArgsConstructor;

@RestController
@RequestMapping("/api/auth")
@RequiredArgsConstructor
public class AuthController {

  private final AuthService authService;
  private final JwtProvider jwtProvider;

  @PostMapping("/signup")
  public ResponseEntity<?> signup(@RequestBody SignupRequest request) {
    authService.signup(request);
    return ResponseEntity.ok("회원가입 성공");
  }

  @PostMapping("/google")
  public ResponseEntity<?> googleLogin(@RequestBody Map<String, String> body) throws Exception {
    String credential = body.get("credential");

    var payload = GoogleAuthService.verify(credential);

    String email = payload.getEmail();
    String name = (String) payload.get("name");
    String picture = (String) payload.get("picture");

    // 회원 존재 확인
    Customers user = authService.findByEmail(email);

    // 없으면 자동 회원가입
    if (user == null) {
      user = authService.googleSignup(email, name, picture);
    }

    // 로그인 처리 → JWT 발급
    String token = jwtProvider.generateToken(user.getCustomersId());

    return ResponseEntity.ok(Map.of("accessToken", token));
  }
}