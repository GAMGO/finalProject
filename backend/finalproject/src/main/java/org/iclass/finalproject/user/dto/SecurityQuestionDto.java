// src/main/java/org/iclass/finalproject/user/dto/SecurityQuestionDto.java
package org.iclass.finalproject.user.dto;

import lombok.Data;
import org.iclass.finalproject.user.model.SecurityQuestionType;

@Data
public class SecurityQuestionDto {
    private SecurityQuestionType question;
    private String answer;
}

/*
 * [파일 설명]
 * - 보안 질문/답변 한 세트를 표현하는 DTO.
 * - 보안 질문 설정/수정 API에서 리스트 형태로 사용.
 */
