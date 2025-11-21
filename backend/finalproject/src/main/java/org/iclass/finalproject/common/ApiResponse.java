package org.iclass.finalproject.common;

public class ApiResponse<T> {

    private final boolean success;
    private final T data;
    private final String message;

    // 기본 생성/정적 팩토리만 쓰게 final + private 생성자
    private ApiResponse(boolean success, T data, String message) {
        this.success = success;
        this.data = data;
        this.message = message;
    }

    public boolean isSuccess() {
        return success;
    }

    public T getData() {
        return data;
    }

    public String getMessage() {
        return message;
    }

    /* ===========================
       ✅ 성공 응답 (ok / success)
       =========================== */

    // data 있는 성공
    public static <T> ApiResponse<T> ok(T data) {
        return new ApiResponse<>(true, data, "success");
    }

    // data 없는 성공(null)
    public static ApiResponse<Void> ok() {
        return new ApiResponse<>(true, null, "success");
    }

    // 컨트롤러에서 success(...) 쓰던 것도 그대로 살림
    public static <T> ApiResponse<T> success(T data) {
        return ok(data);
    }

    public static ApiResponse<Void> success() {
        return ok();
    }

    /* ===========================
       ✅ 에러 응답
       =========================== */

    public static ApiResponse<Void> error(String message) {
        return new ApiResponse<>(false, null, message);
    }

    public static <T> ApiResponse<T> error(String message, T data) {
        return new ApiResponse<>(false, data, message);
    }
}
