package org.iclass.customer.dto;

import lombok.Builder;
import lombok.Getter;

@Getter
@Builder
public class LoginResponse {
    private String token;
    private String refreshToken;
    private String tokenType;
    private String id;

    // ✅ 스샷의 .role(role) 때문에 필요
    private String role;
}
