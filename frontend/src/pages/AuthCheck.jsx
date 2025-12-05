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
        const jsonPayload = decodeURIComponent(atob(base64).split('').map(function(c) {
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
        // ⭐️ 토큰 확인 로직을 함수로 분리
        const checkAuthStatus = () => {
            const token = localStorage.getItem("jwtToken");
            // 🚨 [디버깅 로그 추가] 실제 토큰 값을 콘솔에 출력
            if (token && token.length > 0) {
                console.log(`[DEBUG] sessionStorage에서 토큰 감지: 길이=${token.length}, 값=${token.substring(0, 10)}...`);
            } else if (token === "") {
                console.log("[DEBUG] sessionStorage에서 빈 문자열(\"\") 감지.");
            } else {
                console.log("[DEBUG] sessionStorage에 'jwtToken' 키 없음 (null 반환).");
            }
            let isValid = false;
            
            if (token) {
                try {
                    // jwtDecode 대신 수동 파싱 함수 사용
                    const decoded = manualJwtDecode(token);
                    
                    if (decoded && decoded.exp) {
                        const currentTime = Date.now() / 1000;
                        if (decoded.exp > currentTime) {
                            isValid = true;
                        } else {
                            console.log("Access Token 만료. 토큰 제거.");
                            localStorage.removeItem("jwtToken");
                        }
                    } else {
                        console.error("Access Token 디코딩 실패 또는 exp 필드 누락. 토큰 제거.");
                        localStorage.removeItem("jwtToken");
                    }
                } catch (error) {
                    // 거의 실행되면 안 되는 catch문임.
                    console.error("JWT 디코딩 또는 만료 검사 오류:", error);
                    localStorage.removeItem("jwtToken");
                }
            }else {
                console.log("Storage에 Access Token 없음.");
            }
            setIsAuthenticated(isValid);
            setIsLoading(false);
            if (!isValid) {
                // 현재 URL이 /login이 아니라면 /login으로 리다이렉트
                if (window.location.pathname !== '/login') {
                    navigate('/login', { replace: true });
                }
            }
        };
        
        //초기 상태 확인
        checkAuthStatus();
        
        //storage 이벤트 리스너 추가 (상태 변경 감지)
        const handleStorageChange = () => checkAuthStatus();
        window.addEventListener('storage', handleStorageChange);
        
        return () => {
            window.removeEventListener('storage', handleStorageChange);
        };
    }, [navigate]);

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
    // 3. 인증 성공 → children 렌더링
    return children;
};

export default AuthCheck;