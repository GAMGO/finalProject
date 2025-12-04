import React, { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
// ----------------------------------------------------
// ⭐️ 순수 JavaScript를 이용한 JWT 수동 파싱 함수
// ----------------------------------------------------
const manualJwtDecode = (token) => {
    try {
        // 1. JWT의 페이로드(두 번째 부분)를 가져옵니다.
        const base64Url = token.split('.')[1];
        
        // 2. Base64URL 포맷을 일반 Base64 포맷으로 변환합니다.
        const base64 = base64Url.replace(/-/g, '+').replace(/_/g, '/');
        
        // 3. Base64 디코딩 및 JSON 파싱 (UTF-8 인코딩 처리를 포함)
        const jsonPayload = decodeURIComponent(atob(base64).split('').map(function(c) {
            return '%' + ('00' + c.charCodeAt(0).toString(16)).slice(-2);
        }).join(''));

        return JSON.parse(jsonPayload);
    } catch (e) {
        console.error("JWT 수동 파싱 실패:", e);
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
            const token = sessionStorage.getItem("jwtToken");
            let isValid = false;
            
            if (token) {
                try {
                    // ⭐️ jwtDecode 대신 수동 파싱 함수 사용
                    const decoded = manualJwtDecode(token);
                    
                    if (decoded && decoded.exp) {
                        const currentTime = Date.now() / 1000;
                        if (decoded.exp > currentTime) {
                            isValid = true;
                        } else {
                            sessionStorage.removeItem("jwtToken");
                        }
                    } else {
                        sessionStorage.removeItem("jwtToken");
                    }
                } catch (error) {
                    // 수동 파싱 함수에서 이미 오류 처리를 하지만, 안전을 위해 남겨둡니다.
                    console.error("JWT 디코딩 또는 만료 검사 오류:", error);
                    sessionStorage.removeItem("jwtToken");
                }
            }
            setIsAuthenticated(isValid);
            setIsLoading(false);
        };
        
        //초기 상태 확인
        checkAuthStatus();
        
        //storage 이벤트 리스너 추가 (상태 변경 감지)
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
    // 3. 인증 성공 → children 렌더링
    return children;
};
export default AuthCheck;