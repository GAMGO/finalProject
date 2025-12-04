import React, { useEffect, useState } from "react";
import { Navigate } from "react-router-dom";

// ----------------------------------------------------
// 순수 JavaScript를 이용한 JWT 수동 파싱 함수
// ----------------------------------------------------
const manualJwtDecode = (token) => {
    try {
        // 1. JWT의 페이로드(두 번째 부분)를 가져옵니다.
        const base64Url = token.split('.')[1];
        
        // 2. Base64URL 포맷을 일반 Base64 포맷으로 변환하고 패딩을 추가합니다.
        const base64 = base64Url.replace(/-/g, '+').replace(/_/g, '/');
        const paddedBase64 = base64.padEnd(base64.length + (4 - base64.length % 4) % 4, '=');
        
        // Base64 디코딩 (Latin1 문자열)
        const rawPayload = atob(paddedBase64);
        
        // Latin1 문자열을 UTF-8로 변환하여 JSON 파싱
        const jsonPayload = decodeURIComponent(
            rawPayload.split('').map(c => 
                '%' + ('00' + c.charCodeAt(0).toString(16)).slice(-2)
            ).join('')
        );

        return JSON.parse(jsonPayload);
    } catch (e) {
        console.error("JWT 수동 파싱 실패:", e);
        return null;
    }
};


// ⭐️ 콤포넌트 정의와 동시에 기본 내보내기 (인라인 default export)
//    -> 번들러가 '이 파일은 오직 하나의 함수 컴포넌트'라는 것을 명확히 알게 됨
export default ({ children }) => {
    const [isAuthenticated, setIsAuthenticated] = useState(false);
    const [isLoading, setIsLoading] = useState(true);

    useEffect(() => {
        const checkAuthStatus = () => {
            const token = sessionStorage.getItem("jwtToken");
            let isValid = false;
            
            if (token) {
                try {
                    const decoded = manualJwtDecode(token);
                    
                    if (decoded && decoded.exp) {
                        const currentTime = Date.now() / 1000;
                        if (decoded.exp > currentTime) {
                            isValid = true;
                        } else {
                            // 토큰 만료
                            sessionStorage.removeItem("jwtToken");
                        }
                    } else {
                        // 유효하지 않은 토큰 형식 (exp 필드 없음 등)
                        sessionStorage.removeItem("jwtToken");
                    }
                } catch (error) {
                    // 파싱 실패 또는 기타 오류
                    console.error("JWT 디코딩 또는 만료 검사 오류:", error);
                    sessionStorage.removeItem("jwtToken");
                }
            }
            setIsAuthenticated(isValid);
            setIsLoading(false);
        };
        
        checkAuthStatus();
        
        // storage 이벤트 리스너 추가
        const handleStorageChange = () => checkAuthStatus();
        window.addEventListener('storage', handleStorageChange);
        
        return () => {
            window.removeEventListener('storage', handleStorageChange);
        };
    }, []);

    // 1. 로딩 중일 때
    if (isLoading) {
        return (
            <div className="flex justify-center items-center h-screen bg-gray-100">
                <div className="text-xl font-semibold text-gray-700">
                    인증 상태 확인 중...
                </div>
            </div>
        );
    }

    // 2. 인증 실패 → /login로 리다이렉트
    if (!isAuthenticated) {
        return <Navigate to="/login" replace />;
    }

    // 3. 인증 성공 → children 렌더링
    return children;
};