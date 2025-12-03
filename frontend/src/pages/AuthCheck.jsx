import React, { useEffect, useState } from "react";
import { Navigate } from "react-router-dom";
import { jwtDecode } from 'jwt-decode'

const AuthCheck = ({ children }) => {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    // ⭐️ (개선 3) 토큰 확인 로직을 함수로 분리 (안정성 및 재사용성 확보)
    const checkAuthStatus = () => {
      const token = sessionStorage.getItem("jwtToken");
      let isValid = false;
      if (token) {
        try {
          // ⭐️ (수정 1) token이 존재할 때만, useEffect 내부에서 디코딩합니다.
          const decoded = jwtDecode(token);
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
          // 디코딩 오류 발생 시 토큰 제거
          console.error("JWT 디코딩 오류:", error);
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
    };}, []);

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

export default AuthCheck;
