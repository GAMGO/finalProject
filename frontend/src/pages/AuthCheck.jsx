import React, { useEffect, useState } from "react";
import { Navigate } from "react-router-dom";
//import jwt_decode from 'jwt-decode'; // 실제 프로젝트에서 사용될 라이브러리

// [임시 함수] JWT 토큰의 Payload를 Base64 디코딩하여 JSON 객체로 반환합니다.
// 실제 환경에서는 'jwt-decode'와 같은 라이브러리를 사용해야 안정적입니다.
const decodeJwt = (token) => {
  try {
    // 토큰은 Header.Payload.Signature 형식입니다. Payload(두 번째 부분)를 가져옵니다.
    const base64Url = token.split(".")[1];
    // Base64 URL Safe 형식을 일반 Base64 형식으로 변환합니다.
    const base64 = base64Url.replace(/-/g, "+").replace(/_/g, "/");
    // atob()를 사용해 Base64 디코딩 후, decodeURIComponent로 UTF-8 문자열로 변환합니다.
    const jsonPayload = decodeURIComponent(
      atob(base64)
        .split("")
        .map(function (c) {
          return "%" + ("00" + c.charCodeAt(0).toString(16)).slice(-2);
        })
        .join("")
    );
    return JSON.parse(jsonPayload);
  } catch (e) {
    // 토큰 형식이 잘못되었거나 디코딩에 실패한 경우
    return null;
  }
};

const AuthCheck = ({ children }) => {
    const [isAuthenticated, setIsAuthenticated] = useState(false);
    const [isLoading, setIsLoading] = useState(true); 

    useEffect(() => {
        const token = localStorage.getItem('jwtToken');
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
                    localStorage.removeItem('jwtToken');
                }
            } else {
                localStorage.removeItem('jwtToken');
            }
        }

        // 인증 성공/실패 여부를 상태에 저장
        setIsAuthenticated(isValid); 
        setIsLoading(false);
        
    }, []); // 의존성 배열을 비워 최초 1회만 실행되도록 합니다.

    // 1. 로딩 중일 때
    if (isLoading) {
        return (
            <div className="flex justify-center items-center h-screen bg-gray-100">
                <div className="text-xl font-semibold text-gray-700">인증 상태 확인 중...</div>
            </div>
        );
    }

    // 2. 인증에 실패했을 때 (토큰 없음 또는 만료)
    if (!isAuthenticated) {
        // '/auth' 경로로 리디렉션합니다. (URL은 /auth로 변경됨)
        return <Navigate to="/auth" replace />;
    }

    // 3. 인증에 성공했을 때
    // children으로 전달된 컴포넌트 (<App />)를 렌더링합니다. (URL은 '/'로 유지됨)
    return children;
};

export default AuthCheck;
