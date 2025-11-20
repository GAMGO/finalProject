package org.iclass.finalproject.user.dto;

import jakarta.validation.constraints.Email;
import jakarta.validation.constraints.NotBlank;
import lombok.Data;

@Data
public class LoginRequest {

    @Email @NotBlank
    private String email;

    @NotBlank
    private String password;
}

/*
 * [파일 설명]
 * - 로그인 API 요청 바디 DTO.
 * - 실제 프로젝트에서는 이 정보를 기반으로 인증(JWT/세션)을 처리하게 됨.
 */
