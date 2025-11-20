package org.iclass.finalproject.common;

import lombok.AllArgsConstructor;
import lombok.Data;

@Data
@AllArgsConstructor
public class ApiResponse<T> {
    private boolean success;
    private T data;
    private String message;

    public static <T> ApiResponse<T> ok(T data) {
        return new ApiResponse<>(true, data, null);
    }

    public static <T> ApiResponse<T> error(String message) {
        return new ApiResponse<>(false, null, message);
    }
}

/*
 * [파일 설명]
 * - 모든 REST 응답을 한 형식으로 맞추기 위한 래퍼 클래스.
 * - success/data/message 3가지를 기본으로 사용.
 * - 프론트 팀이 "응답 형식" 통일해서 파싱하기 쉽게 만드는 역할.
 */
