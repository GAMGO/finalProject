// src/pages/AuthCheck.jsx
import React, { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
// ----------------------------------------------------
// ⭐️ 순수 JavaScript를 이용한 JWT 수동 파싱 함수
// ----------------------------------------------------
const manualJwtDecode = (token) => {
    try {
        // 🚨 [수정 1] 토큰이 유효한 문자열이 아니면 즉시 null 반환
        if (!token || typeof token !== 'string' || token.length < 10) {
            console.warn("JWT 수동 파싱 실패: 토큰이 유효하지 않거나 너무 짧습니다.");
            return null;
        }

        // 🚨 [수정 2] 'Bearer ' 접두사를 제거합니다.
        let rawToken = token;
        if (rawToken.startsWith('Bearer ')) {
            rawToken = rawToken.substring(7);
        }

        // 1. JWT의 페이로드(두 번째 부분)를 가져옵니다.
        const parts = rawToken.split('.');

        // 2. 토큰이 '헤더.페이로드.서명' 3부분으로 이루어져 있는지 확인
        if (parts.length !== 3) {
            console.error("JWT 수동 파싱 실패: 토큰 형식이 '헤더.페이로드.서명'이 아닙니다. (분할된 부분 수: " + parts.length + ")");
            return null;
        }

        const base64Url = parts[1];

        // 3. Base64URL 포맷을 일반 Base64 포맷으로 변환합니다.
        const base64 = base64Url.replace(/-/g, '+').replace(/_/g, '/');

        // 4. Base64 디코딩 및 JSON 파싱
        const jsonPayload = decodeURIComponent(atob(base64).split('').map(function (c) {
            return '%' + ('00' + c.charCodeAt(0).toString(16)).slice(-2);
        }).join(''));

        return JSON.parse(jsonPayload);
    } catch (e) {
        console.error("JWT 수동 파싱 중 예기치 않은 오류 발생:", e);
        return null;
    }
};

const AuthCheck = ({ children }) => {
    const navigate = useNavigate();
    const [isAuthenticated, setIsAuthenticated] = useState(false);
    const [isLoading, setIsLoading] = useState(true);

    useEffect(() => {
        // ⭐️ 인증 상태를 확인하고 필요한 경우 리다이렉트하는 함수
        const checkAuthStatus = () => {
            const token = localStorage.getItem("jwtToken");
            let isValid = false;

            if (token) {
                const decoded = manualJwtDecode(token);
                // 토큰이 존재하고 만료 시간이 현재보다 미래인 경우 유효함
                if (decoded && decoded.exp > Date.now() / 1000) {
                    isValid = true;
                }
            }

            setIsAuthenticated(isValid);
            setIsLoading(false);

            const isRecoveryPath = window.location.pathname.startsWith('/recovery');
            const isRootPath = window.location.pathname === '/';

            // 로그인되어 있지 않고, 현재 경로가 /login도 아니고, /recovery도 아니고, 루트 경로('/')도 아닌 경우에만 리다이렉트합니다.
            if (!isValid && window.location.pathname !== '/login' && !isRecoveryPath && !isRootPath) {
                navigate('/login', { replace: true });
            }
        };

        // 1. 컴포넌트 마운트 시 초기 상태 확인
        checkAuthStatus();

        // 2. EmailAuth.jsx에서 dispatchEvent('storage')를 호출하면 즉시 감지하도록 리스너 추가
        window.addEventListener('storage', checkAuthStatus);

        return () => {
            window.removeEventListener('storage', checkAuthStatus);
        };
    }, [navigate]); // navigate 객체 변경 시 재실행

    // 1. 인증 상태 확인 중일 때 로딩 표시
    if (isLoading) {
        return (
            <div className="flex justify-center items-center h-screen bg-gray-100">
                <div className="text-xl font-semibold text-gray-700">
                    인증 상태 확인 중...
                </div>
            </div>
        );
    }

    // 2. 인증 성공 시에만 children(메인 화면) 렌더링
    // 인증 실패 시에는 위 useEffect 내에서 navigate가 작동하므로 여기는 렌더링되지 않음
    return children;
};
export default AuthCheck;