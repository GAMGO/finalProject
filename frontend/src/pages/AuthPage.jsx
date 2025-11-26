// 파일 이름: AuthPage.jsx (또는 상위 컴포넌트)

import React, { useState } from "react";
import LoginPage from "./LoginPage"; // LoginPage 경로를 확인하세요
import SignupPage from "./SignupPage"; // SignupPage 경로를 확인하세요

const AuthPage = () => {
  // 🌟 이 상태가 로그인/회원가입 모드를 결정합니다.
  const [isLoginMode, setIsLoginMode] = useState(true);

  // 🌟 이 함수를 자식 컴포넌트(LoginPage, SignupPage)에 onToggleMode로 전달합니다.
  const toggleMode = () => {
    setIsLoginMode((prev) => !prev);
  };

  return (
    <>
      {/* 🌟 isLoginMode 값에 따라 조건부 렌더링 및 프롭 전달 */}
      {
        isLoginMode ? (
          <LoginPage onToggleMode={toggleMode} key="login" /> // key 추가
        ) : (
          <SignupPage onToggleMode={toggleMode} key="signup" />
        ) // key 추가
      }
    </>
  );
};

export default AuthPage;
