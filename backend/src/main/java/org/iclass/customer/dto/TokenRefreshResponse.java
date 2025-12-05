package org.iclass.customer.dto;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class TokenRefreshResponse {
    private String token;
    private String refreshToken; // 재발급된 Access Token과 함께 기존 Refresh Token을 반환
    private String tokenType; // "Bearer"
}