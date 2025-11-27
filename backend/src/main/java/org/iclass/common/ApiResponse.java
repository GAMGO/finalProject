package org.iclass.common;

import lombok.AllArgsConstructor;
import lombok.Data;

@Data
@AllArgsConstructor
public class ApiResponse<T> {

    private boolean success;
    private T data;
    private String message;

    // =========================
    //  기존 팩토리 메소드
    // =========================

    /** 성공 + data */
    public static <T> ApiResponse<T> ok(T data) {
        return new ApiResponse<>(true, data, null);
    }

    /** 성공 + data + message */
    public static <T> ApiResponse<T> ok(T data, String message) {
        return new ApiResponse<>(true, data, message);
    }

    /** 실패 + message */
    public static <T> ApiResponse<T> error(String message) {
        return new ApiResponse<>(false, null, message);
    }

    /** 실패 + data + message */
    public static <T> ApiResponse<T> error(T data, String message) {
        return new ApiResponse<>(false, data, message);
    }

    // =========================
    //  컨트롤러에서 쓰는 별칭들
    // =========================

    /** ok() 별칭 – 컨트롤러에서 success()로 사용 가능 */
    public static <T> ApiResponse<T> success(T data) {
        return ok(data);
    }

    public static <T> ApiResponse<T> success(T data, String message) {
        return ok(data, message);
    }

    /** error() 별칭 – 필요하면 fail()도 사용 가능 */
    public static <T> ApiResponse<T> fail(String message) {
        return error(message);
    }

    public static <T> ApiResponse<T> fail(T data, String message) {
        return error(data, message);
    }
}

/*
 * [파일 설명]
 * - 모든 REST 응답을 한 형식으로 맞추기 위한 래퍼 클래스.
 * - success/data/message 3가지를 기본으로 사용.
 * - ok()/error()와 success()/fail() 둘 다 제공해서,
 *   팀원들이 편한 메소드 이름 골라 쓸 수 있게 함.
 */