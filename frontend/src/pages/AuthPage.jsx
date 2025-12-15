// src/pages/AuthPage.jsx
import React, { useState } from "react";
import LoginPage from "./LoginPage";
import SignupPage from "./SignupPage";
import EmailAuth from "./EmailAuth";
import WithdrawalPage from "./WithdrawalPage";
import RecoveringPage from "./RecoveringPage";
import FindAccountPage from "./FindAccountPage";

const AuthPage = ({ onLoginSuccess }) => {
  const initialMode = window.location.pathname.startsWith('/recovery') ? 'recover' : 'login';
  const [authMode, setAuthMode] = useState(initialMode);
  const [registeredEmail, setRegisteredEmail] = useState('');
  const [userIdForWithdrawal, setUserIdForWithdrawal] = useState("");

  const toggleMode = () => {
    setAuthMode((prev) => (prev === "login" ? "signup" : "login"));
  };
  const [signupPayload, setSignupPayload] = useState(null);

  const handleSignupSuccess = async (email, payload) => {
    setRegisteredEmail(email);
    setSignupPayload(payload);
    setAuthMode('EmailAuth');
  };

  const handleAuthSuccess = () => {
    onLoginSuccess();
  };

  const renderContent = () => {
    switch (authMode) {
      case 'login':
        return (
          <LoginPage
            onToggleMode={toggleMode}
            onLoginSuccess={onLoginSuccess}
            onWithdrawMode={(id) => {
              setUserIdForWithdrawal(id);
              setAuthMode('withdrawal');
            }}
            onFindPassword={() => setAuthMode('findPasswordAccount')}
            onFindId={() => setAuthMode('findIdAccount')}
          />
        );
      case 'withdrawal':
        return (
          <WithdrawalPage
            userId={userIdForWithdrawal}
            onLogout={() => setAuthMode('login')}
          />
        );
      case 'recover':
        return (
          <RecoveringPage
            onAuthSuccess={() => setAuthMode('login')}
          />
        );
      case 'signup':
        return (
          <SignupPage
            onToggleMode={toggleMode}
            onSignupSuccess={handleSignupSuccess}
            key="signup"
          />
        );
      case 'EmailAuth':
        return (
          <EmailAuth
            registeredEmail={registeredEmail}
            signupPayload={signupPayload}
            onAuthSuccess={handleAuthSuccess}
            onRestartSignup={() => setAuthMode('signup')}
          />
        );
      case 'findIdAccount':
        return (
          <FindAccountPage
            onGoToLogin={() => setAuthMode('login')}
            initialMode={'findId'} // 아이디 찾기 모드로 시작
          />
        );
      case 'findPasswordAccount':
        return (
          <FindAccountPage
            onGoToLogin={() => setAuthMode('login')}
            initialMode={'findPassword'} // 비밀번호 찾기 모드로 시작
          />
        );
      default:
        return null;
    }
  }

  const containerStyle = {
    display: "flex",
    justifyContent: "center",
    alignItems: "center",
    width: "100vw",
    height: "100vh",
    backgroundColor: "#78266A", // 다크 퍼플 배경
  };

  return (
    <div style={containerStyle}>
      {renderContent()}
    </div>
  );
};

export default AuthPage;