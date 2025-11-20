package org.iclass.finalproject.common;

import jakarta.servlet.http.HttpSession;
import org.springframework.web.context.request.RequestContextHolder;
import org.springframework.web.context.request.ServletRequestAttributes;

public class CurrentUser {

    public static final String SESSION_KEY = "USER_ID";

    public static Long getUserId() {
        ServletRequestAttributes attrs =
                (ServletRequestAttributes) RequestContextHolder.getRequestAttributes();

        if (attrs == null) {
            throw new IllegalStateException("요청 컨텍스트가 없습니다. 로그인 상태가 아닐 수 있습니다.");
        }

        HttpSession session = attrs.getRequest().getSession(false);
        if (session == null) {
            throw new IllegalStateException("로그인 세션이 없습니다. 다시 로그인해주세요.");
        }

        Object val = session.getAttribute(SESSION_KEY);
        if (val == null) {
            throw new IllegalStateException("세션에 사용자 정보가 없습니다. 다시 로그인해주세요.");
        }

        if (val instanceof Long l) return l;
        if (val instanceof Integer i) return i.longValue();
        return Long.valueOf(val.toString());
    }
}

/*
 * [파일 설명]
 * - 현재 HTTP 요청에 연결된 세션에서 userId를 꺼내오는 헬퍼 클래스.
 * - AuthController.login()에서 세션에 USER_ID를 저장해두면,
 *   이후 모든 요청은 CurrentUser.getUserId()로 로그인한 유저 id를 얻어 사용.
 */
