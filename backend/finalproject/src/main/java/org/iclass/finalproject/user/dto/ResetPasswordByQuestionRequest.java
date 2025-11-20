// src/main/java/org/iclass/finalproject/user/dto/ResetPasswordByQuestionRequest.java
package org.iclass.finalproject.user.dto;

import jakarta.validation.constraints.Email;
import jakarta.validation.constraints.NotBlank;
import lombok.Data;
import org.iclass.finalproject.user.model.SecurityQuestionType;

@Data
public class ResetPasswordByQuestionRequest {

    @Email @NotBlank
    private String email;

    private SecurityQuestionType question;

    @NotBlank
    private String answer;

    @NotBlank
    private String newPassword;
}

/*
 * [파일 설명]
 * - 비밀번호를 잊었을 때 "이메일 + 보안질문"으로 비밀번호 재설정하는 요청 DTO.
 * - 실제 이메일 인증 토큰은 별도 구현(지금은 이메일을 ID처럼 사용).
 */
