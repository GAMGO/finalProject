import React, { useState } from "react";
import { useNavigate } from 'react-router-dom';
import LoginPage from "./LoginPage";
import SignupPage from "./SignupPage";
import EmailAuth from "./EmailAuth";
const baseURL = import.meta.env.VITE_BASE_URL;
const AuthPage = ({ onLoginSuccess }) => {
  const navigate = useNavigate();
  // 🌟 이 상태가 로그인/회원가입 모드를 결정합니다.
  const [authMode, setAuthMode] = useState('login');
  const [registeredEmail, setRegisteredEmail] = useState('');
  // 🌟 이 함수를 자식 컴포넌트(LoginPage, SignupPage)에 onToggleMode로 전달합니다.
  const toggleMode = () => {
    setAuthMode(prev => prev === 'login' ? 'signup' : 'login');
  };
  //회원가입 성공 시 호출될 함수 (Signup -> EmailAuth 전환)
  const handleSignupSuccess = (email) => {
    setRegisteredEmail(email); // 이메일 저장
    setAuthMode('EmailAuth');  // 모드를 'emailAuth'로 변경
  };
  
  //인증 성공 시 호출될 함수 (EmailAuth -> login 전환)
  const handleAuthSuccess = () => {
    setAuthMode('login'); 
    setRegisteredEmail('');
    navigate('/')
  };
  const renderContent = () => {
      switch (authMode) {
          case 'login':
              // LoginPage가 onToggleMode를 통해 signup으로 이동합니다.
              return <LoginPage onToggleMode={toggleMode} onLoginSuccess={onLoginSuccess} key="login" />;
          case 'signup':
              return (
                  // SignupPage에 다음 단계 전환 함수 전달
                  <SignupPage 
                      onToggleMode={toggleMode} // login <-> Signup 전환
                      onSignupSuccess={handleSignupSuccess} // ⭐️ EmailAuth 전환용
                      key="signup"
                  />
              );
          case 'EmailAuth':
              return (
                  //EmailAuth 렌더링 및 데이터/콜백 전달
                  <EmailAuth
                      registeredEmail={registeredEmail} // ⭐️ 전달받은 이메일
                      onAuthSuccess={handleAuthSuccess} // 인증 성공 시 login으로 전환
                      onRestartSignup={toggleMode} // 필요하다면 Signup으로 돌아가기 (로직에 따라 toggleMode 사용 가능)
                      key="EmailAuth"
                  />
              );
          default:
              return null;
      }
  }
  return (
      <>
        {renderContent()}
      </>
  );
};

export default AuthPage;
