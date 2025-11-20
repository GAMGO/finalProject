// src/main/java/org/iclass/finalproject/user/dto/SetSecurityQuestionsRequest.java
package org.iclass.finalproject.user.dto;

import jakarta.validation.constraints.NotEmpty;
import lombok.Data;

import java.util.List;

@Data
public class SetSecurityQuestionsRequest {

    @NotEmpty
    private List<SecurityQuestionDto> questions;
}

/*
 * [파일 설명]
 * - 유저가 보안 질문들을 설정하거나 수정할 때 사용하는 요청 DTO.
 */
