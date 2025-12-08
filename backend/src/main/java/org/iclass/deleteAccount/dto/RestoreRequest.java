package org.iclass.deleteAccount.dto;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
public class RestoreRequest {

    @Schema(description = "이메일로 발송된 6자리 복구 토큰", example = "123456")
    private String recoveryToken;
}