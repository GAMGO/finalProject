import React, { useEffect, useState } from "react";
import { Navigate } from "react-router-dom";
//import jwt_decode from 'jwt-decode'; // 실제 프로젝트에서 사용될 라이브러리

// [임시 함수] JWT 토큰의 Payload를 Base64 디코딩하여 JSON 객체로 반환합니다.
// 실제 환경에서는 'jwt-decode'와 같은 라이브러리를 사용해야 안정적입니다.
const decodeJwt = (token) => {
  try {
    const base64Url = token.split(".")[1];
    const base64 = base64Url.replace(/-/g, "+").replace(/_/g, "/");
    const jsonPayload = decodeURIComponent(
      atob(base64)
        .split("")
        .map((c) => "%" + ("00" + c.charCodeAt(0).toString(16)).slice(-2))
        .join("")
    );
    return JSON.parse(jsonPayload);
  } catch (e) {
    return null;
  }
};

const AuthCheck = ({ children }) => {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const token = localStorage.getItem("jwtToken");
    let isValid = false;

    if (token) {
      const decoded = decodeJwt(token);

      if (decoded && decoded.exp) {
        const currentTime = Date.now() / 1000;

        if (decoded.exp > currentTime) {
          isValid = true;
          console.log("JWT 토큰 만료 시간 확인: 유효함");
        } else {
          console.log("JWT 토큰 만료 시간 확인: 만료됨. 토큰 삭제.");
          localStorage.removeItem("jwtToken");
        }
      } else {
        localStorage.removeItem("jwtToken");
      }
    }

    setIsAuthenticated(isValid);
    setIsLoading(false);
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

  // 2. 인증 실패 → /auth로 리다이렉트
  if (!isAuthenticated) {
    return <Navigate to="/login" replace />;
  }

  // 3. 인증 성공 → children 렌더링
  return children;
};

export default AuthCheck;
