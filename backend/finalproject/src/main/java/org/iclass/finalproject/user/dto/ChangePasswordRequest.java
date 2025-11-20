// src/main/java/org/iclass/finalproject/user/dto/ChangePasswordRequest.java
package org.iclass.finalproject.user.dto;

import jakarta.validation.constraints.NotBlank;
import lombok.Data;

@Data
public class ChangePasswordRequest {

    @NotBlank
    private String currentPassword;

    @NotBlank
    private String newPassword;
}

/*
 * [파일 설명]
 * - "현재 비밀번호를 알고 있을 때" 비밀번호 변경 API 요청 바디.
 */
