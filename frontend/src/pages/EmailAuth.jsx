import React, { useState, useEffect } from "react";
import axios from "axios";

const baseURL = import.meta.env.VITE_LOCAL_BASE_URL;

const EmailAuthPage = ({ registeredEmail, signupPayload, onAuthSuccess, onRestartSignup }) => {
  const [authCode, setAuthCode] = useState("");
  const [countdown, setCountdown] = useState(300);
  const [isVerifying, setIsVerifying] = useState(false);
  const [isResending, setIsResending] = useState(false);
  const [message, setMessage] = useState({ text: "", type: "" });

  // 1. 폰트 및 스타일 정의 (LoginPage 기반 원본 스타일 복구)
  const darkPurple = "#78266A";
  const lightPeach = "#F5D7B7";
  const white = "#FFFFFF";
  const customFont = "PartialSans, SchoolSafetyRoundedSmile, sans-serif";
  const clearCustomFont = "SchoolSafetyRoundedSmile, sans-serif";

  const fontFaceCss = `
    @font-face { font-family: 'PartialSans'; src: url('https://cdn.jsdelivr.net/gh/projectnoonnu/noonfonts_2307-1@1.1/PartialSansKR-Regular.woff2') format('woff2'); }
    @font-face { font-family: 'SchoolSafetyRoundedSmile'; src: url('https://cdn.jsdelivr.net/gh/projectnoonnu/2408-5@1.0/HakgyoansimDunggeunmisoTTF-R.woff2') format('woff2'); }
  `;

  // 컴포넌트 폭발 방지를 위한 스타일 객체 선언 확인
  const containerStyle = {
    display: "flex",
    justifyContent: "center",
    alignItems: "center",
    width: "100vw",
    height: "100vh",
    backgroundColor: darkPurple,
    fontFamily: customFont,
  };

  const boxStyle = {
    backgroundColor: lightPeach,
    padding: "60px 40px",
    borderRadius: "40px",
    boxShadow: "0 4px 15px rgba(0, 0, 0, 0.3)",
    width: "45vh",
    textAlign: "center",
  };
  // 2. 타이머 및 비즈니스 로직
  useEffect(() => {
    if (countdown <= 0) return;
    const interval = setInterval(() => setCountdown((prev) => prev - 1), 1000);
    return () => clearInterval(interval);
  }, [countdown]);

  const formatTime = (seconds) => {
    const min = Math.floor(seconds / 60);
    const sec = seconds % 60;
    return `${min}:${sec < 10 ? "0" : ""}${sec}`;
  };

const handleResendCode = async () => {
    if (isResending) return;
    setIsResending(true);
    try {
        // ⭐️ 주소창 직접 조립 (encodeURIComponent 사용)
        const resendUrl = `${baseURL}/api/email/resend?email=${encodeURIComponent(registeredEmail)}`;
        await axios.post(resendUrl, null, { withCredentials: true });

        setCountdown(300);
        setMessage("인증 이메일이 발송되었습니다.");
    } catch (error) {
        // 백엔드가 반환하는 400 에러 메시지를 사용자에게 보여줍니다.
        setMessage(error.response?.data?.message || "재발송 실패");
    } finally {
        setIsResending(false);
    }
};

const handleVerifyCode = async () => {
  if (isVerifying) return;
  if (authCode.length !== 6) return setMessage({ text: "6자리 코드를 입력하세요.", type: "error" });
  
  setIsVerifying(true);
  try {
    await axios.post(`${baseURL}/api/email/verify`, {
      email: registeredEmail,
      token: authCode 
    });

    setMessage({ text: "인증 성공! 계정을 생성 중입니다...", type: "success" });
    const res = await axios.post(`${baseURL}/api/auth/signup`, signupPayload);
    
    if (res.data.token) {
      localStorage.setItem("jwtToken", res.data.token);
      localStorage.setItem("refreshToken", res.data.refreshToken);
      onAuthSuccess(); // App.jsx로 진입
    }
  } catch (error) {
    setMessage({ text: "인증 실패 또는 중복 가입 문제입니다.", type: "error" });
  } finally {
    setIsVerifying(false);
  }
};

  // 3. 렌더링 (구조 수정 금지)
  return (
    <div style={containerStyle}>
      <style>{fontFaceCss}</style>
      <div style={boxStyle}>
        <h2 style={{ color: darkPurple, fontSize: "32px", marginBottom: "10px" }}>이메일 인증</h2>
        <p style={{ color: darkPurple, fontSize: "14px", marginBottom: "20px", fontFamily: clearCustomFont }}>
          <b>{registeredEmail}</b>로 인증번호를 보냈습니다.
        </p>

        <div style={{ fontSize: "28px", fontWeight: "bold", marginBottom: "15px", color: countdown <= 30 ? "red" : darkPurple }}>
          {formatTime(countdown)}
        </div>

        <input
          type="text"
          maxLength={6}
          placeholder="000000"
          style={{ width: "100%", padding: "15px", borderRadius: "20px", border: "none", textAlign: "center", fontSize: "24px", letterSpacing: "8px", boxShadow: `4px 4px 0px ${darkPurple}`, outline: "none", marginBottom: "20px" }}
          value={authCode}
          onChange={(e) => setAuthCode(e.target.value.replace(/[^0-9]/g, ""))}
        />

        <button
          style={{ width: "100%", padding: "15px", borderRadius: "20px", border: `2px solid ${darkPurple}`, backgroundColor: white, color: darkPurple, fontWeight: "bold", cursor: "pointer", boxShadow: `4px 4px 0px ${darkPurple}` }}
          onClick={handleVerifyCode}
          disabled={isVerifying}
        >
          {isVerifying ? "가입 처리 중..." : "인증하고 로그인"}
        </button>

        <button
          style={{ background: "none", border: "none", color: darkPurple, textDecoration: "underline", marginTop: "15px", cursor: "pointer", fontSize: "14px", fontFamily: clearCustomFont }}
          onClick={handleResendCode}
          disabled={isResending}
        >
          {isResending ? "재발송 중..." : "코드 재발송하기"}
        </button>

        <button
          style={{ background: "none", border: "none", color: darkPurple, marginTop: "10px", cursor: "pointer", fontSize: "14px", fontFamily: clearCustomFont }}
          onClick={onRestartSignup}
        >
          가입 다시 시작하기
        </button>

        {message.text && (
          <div style={{ marginTop: "15px", padding: "10px", borderRadius: "10px", backgroundColor: message.type === "error" ? "#D9534F" : message.type === "success" ? "#5CB85C" : "#007BFF", color: "white", fontSize: "13px" }}>
            {message.text}
          </div>
        )}
      </div>
    </div>
  );
};

export default EmailAuthPage;