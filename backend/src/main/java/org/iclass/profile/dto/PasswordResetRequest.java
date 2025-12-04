// src/main/java/org/iclass/profile/dto/PasswordResetRequest.java
package org.iclass.profile.dto;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Size;
import lombok.Data;

@Data
public class PasswordResetRequest {

    @NotBlank
    private String code;    // 이메일로 받은 6자리 인증 코드

    @NotBlank
    @Size(min = 8, max = 72)
    private String newPassword;
}
