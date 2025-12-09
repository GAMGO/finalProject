import React, { useState } from "react";
import LoginPage from "./LoginPage";
import SignupPage from "./SignupPage";
import EmailAuth from "./EmailAuth";
import WithdrawalPage from "./WithdrawalPage";
import RecoveringPage from "./RecoveringPage";

const AuthPage = ({ onLoginSuccess }) => {
  const [authMode, setAuthMode] = useState('login');
  const [registeredEmail, setRegisteredEmail] = useState('');
  const [userIdForWithdrawal, setUserIdForWithdrawal] = useState("");

  const toggleMode = () => {
    setAuthMode(prev => prev === 'login' ? 'signup' : 'login');
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