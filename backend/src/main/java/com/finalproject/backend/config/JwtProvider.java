package com.finalproject.backend.config;

import io.jsonwebtoken.*;
import io.jsonwebtoken.security.Keys;
import org.springframework.stereotype.Component;
import java.security.Key;
import java.util.Date;

@Component // ⭐ 스프링 빈 등록 필수
public class JwtProvider {

  private final Key key;

  public JwtProvider() {
    // ⭐ 실제 서비스에서는 환경변수로 주입
    String secretKey = "ThisIsASecretKeyForJwtTokenWhichShouldBeLongEnough123!";
    this.key = Keys.hmacShaKeyFor(secretKey.getBytes());
  }

  // JWT 생성
  public String generateToken(String username) {
    long now = System.currentTimeMillis();
    long expiry = 1000L * 60 * 60 * 24; // 24시간

    return Jwts.builder()
        .setSubject(username)
        .setIssuedAt(new Date(now))
        .setExpiration(new Date(now + expiry))
        .signWith(key, SignatureAlgorithm.HS256)
        .compact();
  }

  // JWT 검증
  public Claims validateToken(String token) {
    return Jwts.parserBuilder()
        .setSigningKey(key)
        .build()
        .parseClaimsJws(token)
        .getBody();
  }

  // username 가져오기
  public String getUsernameFromToken(String token) {
    return validateToken(token).getSubject();
  }
}