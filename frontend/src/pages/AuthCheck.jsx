import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
// import jwt_decode from 'jwt-decode'; // 실제 프로젝트에서 사용될 라이브러리
// import axios from 'axios'; 

// [임시 함수] JWT 토큰의 Payload를 Base64 디코딩하여 JSON 객체로 반환합니다.
// 실제 환경에서는 'jwt-decode'와 같은 라이브러리를 사용해야 안정적입니다.
const decodeJwt = (token) => {
    try {
        // 토큰은 Header.Payload.Signature 형식입니다. Payload(두 번째 부분)를 가져옵니다.
        const base64Url = token.split('.')[1];
        // Base64 URL Safe 형식을 일반 Base64 형식으로 변환합니다.
        const base64 = base64Url.replace(/-/g, '+').replace(/_/g, '/');
        // atob()를 사용해 Base64 디코딩 후, decodeURIComponent로 UTF-8 문자열로 변환합니다.
        const jsonPayload = decodeURIComponent(atob(base64).split('').map(function(c) {
            return '%' + ('00' + c.charCodeAt(0).toString(16)).slice(-2);
        }).join(''));
        return JSON.parse(jsonPayload);
    } catch (e) {
        // 토큰 형식이 잘못되었거나 디코딩에 실패한 경우
        return null;
    }
};

const AuthCheck = () => {
    // 프론트엔드에서 처리하지만, 리디렉션 전 잠시 상태를 보여주기 위해 유지합니다.
    const [isLoading, setIsLoading] = useState(true); 
    const navigate = useNavigate();

    useEffect(() => {
        const checkAuthStatus = () => {
            const token = localStorage.getItem('jwtToken');
            let isValid = false;

            if (token) {
                const decoded = decodeJwt(token); 
                
                if (decoded && decoded.exp) {
                    // exp는 만료 시간 (Unix Time, 초 단위)입니다.
                    const currentTime = Date.now() / 1000; // 현재 시간 (초 단위)
                    
                    if (decoded.exp > currentTime) {
                        // ⭐️ 만료 시간이 현재 시간보다 크면 유효합니다.
                        isValid = true;
                        console.log("JWT 토큰 만료 시간 확인: 유효함");
                    } else {
                        // ⭐️ 만료 시간이 지났다면 토큰을 무효화합니다.
                        console.log("JWT 토큰 만료 시간 확인: 만료됨. 토큰을 삭제하고 로그인 페이지로 리디렉션합니다.");
                        localStorage.removeItem('jwtToken');
                    }
                } else {
                    // 토큰은 있지만 유효하지 않은 형식이라면 삭제합니다. (exp가 없거나 디코딩 실패)
                     localStorage.removeItem('jwtToken');
                }
            }

            if (isValid) {
                // 토큰이 유효하면 메인 앱 페이지로 이동
                navigate('/app', { replace: true });
            } else {
                // 토큰이 없거나 만료되었으면 인증 페이지로 리디렉션
                navigate('/auth', { replace: true });
            }
            
            setIsLoading(false);
        };

        checkAuthStatus();
    }, [navigate]);

    if (isLoading) {
        return (
            <div className="flex justify-center items-center h-screen bg-gray-100">
                <div className="text-xl font-semibold text-gray-700">인증 상태 확인 중...</div>
            </div>
        );
    }

    // 리디렉션이 완료된 후에는 null을 반환합니다.
    return null;
};

export default AuthCheck;