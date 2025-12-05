package org.iclass.security;

import java.time.*;
import java.util.Date;
import javax.crypto.SecretKey;
import org.hibernate.validator.internal.util.stereotypes.Lazy;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.stereotype.Component;
import org.iclass.customer.service.CustomersService;
import io.jsonwebtoken.*;
import io.jsonwebtoken.security.Keys;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;

@Component
@Slf4j
@RequiredArgsConstructor
public class JwtTokenProvider {

    // 🔑 상수 정의: JWT 서명에 사용할 시크릿 키 문자열입니다. (64자리, 256비트 이상)
    private static final String SECRET_KEY_STRING = "756be4cf9581add13ddb3ab3e2f1e75f27a0661af1c1225a89ef9a1d44d3f03b";
    
    // ✨ 수정 1: 시크릿 키를 클래스 필드에서 한 번만 생성하여 메모리에 저장하고 final로 선언합니다.
    private final SecretKey secretKey = Keys.hmacShaKeyFor(SECRET_KEY_STRING.getBytes());
    
    // 토큰 만료 시간 (밀리초 단위, 현재는 24시간)
    private final int jwtExpirationInMs = 24 * 60 * 60 * 1000;
    // Refresh Token 만료 시간 (밀리초 단위, 7일)
    private final int refreshTokenExpirationInMs = 7 * 24 * 60 * 60 * 1000; // 7일
    @Lazy
    private final CustomersService cds;
    public String createToken(Authentication authentication) {
        UserDetails userPrincipal = (UserDetails) authentication.getPrincipal();
        // 토큰 생성 시점 : 현재시간 + 만료시간 = 토큰 만료 시점 
        Date expiryDate = new Date(System.currentTimeMillis() + jwtExpirationInMs); 
        return Jwts.builder()
                .signWith(this.secretKey) // ✨ 수정 2: 초기화된 secretKey 사용, 알고리즘 자동 추론 (HS512)
                .subject(userPrincipal.getUsername()) // 여기서부터 토큰과 관련된 정보 저장
                .issuer("com.example") // 발급자:서비스이름
                .issuedAt(new Date()) // 발급날짜
                .expiration(expiryDate) // 만료날짜
                .compact();
    }
    public String createRefreshToken(Authentication authentication) {
        UserDetails userPrincipal = (UserDetails) authentication.getPrincipal();
        Date expiryDate = new Date(System.currentTimeMillis() + refreshTokenExpirationInMs);

        return Jwts.builder()
                .signWith(this.secretKey)
                .subject(userPrincipal.getUsername())
                .issuer("com.example")
                .issuedAt(new Date())
                .expiration(expiryDate)
                .claim("typ", "REFRESH") // Refresh 토큰임을 명시적으로 표시
                .compact();
    }
    public Authentication getAuthentication(String token) {
        String username = getUsernameFromToken(token);
        UserDetails userDetails = cds.loadUserByUsername(username);
        return new UsernamePasswordAuthenticationToken(userDetails, "", userDetails.getAuthorities());
    }

    // 클라이언트가 보낸 토큰(메소드 인자 String token)을 검증하는 메소드
    public String getUsernameFromToken(String token) {
        Claims claims = Jwts.parser()
                .verifyWith(this.secretKey).build().parseSignedClaims(token).getPayload();
        // subject 는 username를 저장했으므로 토큰 값을 분해해서 얻은 subject 는 username 이다.
        return claims.getSubject();
    }
    // 로그아웃 추가 코드
    public boolean validateToken(String authToken) {
        try {
            Jwts.parser().verifyWith(this.secretKey).build().parseSignedClaims(authToken);
            return true;
        } catch (MalformedJwtException ex) {
            log.error("Invalid JWT token");
        } catch (ExpiredJwtException ex) {
            log.error("Expired JWT token");
        } catch (UnsupportedJwtException ex) {
            log.error("Unsupported JWT token");
        } catch (IllegalArgumentException ex) {
            log.error("JWT claims string is empty");
        } catch (Exception ex) {
            log.error("JWT validation error");
        }
        return false;
    }

    public String getUsername(String token) {
        try {
            return Jwts.parser()
                    .verifyWith(this.secretKey)
                    .build()
                    .parseSignedClaims(token)
                    .getPayload()
                    .getSubject();
        } catch (Exception e) {
            return null;
        }
    }
    public LocalDateTime getExpiry(String token) {
        try {
            Date exp = Jwts.parser()
                    .verifyWith(this.secretKey)
                    .build()
                    .parseSignedClaims(token)
                    .getPayload()
                    .getExpiration();
            return exp == null ? null
                    : exp.toInstant()
                            .atZone(ZoneId.systemDefault())
                            .toLocalDateTime();
        } catch (Exception e) {
            return null;
        }
        //
    }
    // >>> [ADDED] (비밀번호 재설정용) 15분짜리 단기 토큰 발급
    public String createRecoveryToken(String id) {
        Date expiry = new Date(System.currentTimeMillis() + 15 * 60 * 1000);
        return Jwts.builder()
                .signWith(this.secretKey)
                .subject(id)
                .issuer("com.example")
                .issuedAt(new Date())
                .expiration(expiry)
                .claim("typ", "PW_RESET")
                .compact();
    }

    // >>> [ADDED] 복구 토큰 검증 + 사용자ID 추출 (유형 체크)
    public String validateAndGetUserFromRecoveryToken(String token) {
        try {
            Claims c = Jwts.parser().verifyWith(this.secretKey).build()
                    .parseSignedClaims(token).getPayload();
            if (!"PW_RESET".equals(c.get("typ")))
                return null;
            return c.getSubject();
        } catch (Exception e) {
            return null;
        }
    }
}