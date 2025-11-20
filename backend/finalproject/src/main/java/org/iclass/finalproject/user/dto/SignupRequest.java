package org.iclass.finalproject.user.dto;

import jakarta.validation.constraints.Email;
import jakarta.validation.constraints.NotBlank;
import lombok.Data;

@Data
public class SignupRequest {

    @NotBlank
    private String username;

    @Email @NotBlank
    private String email;

    @NotBlank
    private String password;
}

/*
 * [파일 설명]
 * - 회원가입 API 요청 바디를 표현하는 DTO.
 * - 컨트롤러에서 @Valid 붙여 검증하고 서비스에 전달해서 User 엔티티 생성에 사용.
 */
