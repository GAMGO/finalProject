import React, { useState, useEffect } from "react";
import axios from "axios";

const baseURL = import.meta.env.VITE_LOCAL_BASE_URL;

const EmailAuthPage = ({ registeredEmail, onAuthSuccess, onRestartSignup }) => {
  const [authCode, setAuthCode] = useState("");
  const [countdown, setCountdown] = useState(300);
  const [isVerifying, setIsVerifying] = useState(false);
  const [isResending, setIsResending] = useState(false);
  const [message, setMessage] = useState("");

  // ----------------------------------------------------------------------
  // 1. 폰트 및 스타일 정의 (LoginPage 기반)
  // ----------------------------------------------------------------------
  const darkPurple = "#78266A";
  const lightPeach = "#F5D7B7";
  const white = "#FFFFFF";

  const customFont = "PartialSans, SchoolSafetyRoundedSmile, sans-serif";
  const clearCustomFont = "SchoolSafetyRoundedSmile, sans-serif";

  const fontFaceCss = `
    @font-face {
      font-family: 'PartialSans';
      src: url('https://cdn.jsdelivr.net/gh/projectnoonnu/noonfonts_2307-1@1.1/PartialSansKR-Regular.woff2') format('woff2');
      font-weight: normal;
      font-display: swap;
    }
    @font-face {
      font-family: 'SchoolSafetyRoundedSmile';
      src: url('https://cdn.jsdelivr.net/gh/projectnoonnu/2408-5@1.0/HakgyoansimDunggeunmisoTTF-R.woff2') format('woff2');
      font-weight: normal;
      font-display: swap;
    }
    @font-face {
      font-family: 'SchoolSafetyRoundedSmile';
      src: url('https://cdn.jsdelivr.net/gh/projectnoonnu/2408-5@1.0/HakgyoansimDunggeunmisoTTF-B.woff2') format('woff2');
      font-weight: 700;
      font-display: swap;
    }
  `;

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

  // ----------------------------------------------------------------------
  // 2. 비즈니스 로직 (타이머 & 인증)
  // ----------------------------------------------------------------------
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

  const handleVerifyCode = async () => {
    if (authCode.length !== 6) {
      setMessage("6자리 인증번호를 정확히 입력해주세요.");
      return;
    }
    setIsVerifying(true);
    try {
      await axios.post(`${baseURL}/api/email/verify`, {
        email: registeredEmail,
        token: authCode,
      });
      onAuthSuccess();
    } catch (error) {
      setMessage("인증번호가 틀렸거나 만료되었습니다.");
    } finally {
      setIsVerifying(false);
    }
  };

  // ----------------------------------------------------------------------
  // 3. 렌더링
  // ----------------------------------------------------------------------
  return (
    <div style={containerStyle}>
      <style>{fontFaceCss}</style>
      <div style={boxStyle}>
        {/* 제목: PartialSans (customFont) */}
        <h2 style={{ color: darkPurple, fontSize: "32px", marginBottom: "10px", fontFamily: customFont }}>
          이메일 인증
        </h2>
        
        {/* 설명: SchoolSafetyRoundedSmile (clearCustomFont) */}
        <p style={{ color: darkPurple, fontSize: "14px", marginBottom: "20px", lineHeight: "1.5", fontFamily: clearCustomFont }}>
          회원님 이메일 <br />
          <b>{registeredEmail}</b>로 <br />
          인증번호를 발송했습니다.
        </p>

        {/* 타이머: PartialSans (큰 글씨) */}
        <div style={{
          fontSize: "42px",
          fontWeight: "bold",
          fontFamily: customFont,
          color: countdown <= 30 ? "#D9534F" : darkPurple,
          marginBottom: "10px"
        }}>
          {formatTime(countdown)}
        </div>

        {/* 입력란: SchoolSafetyRoundedSmile */}
        <input
          type="text"
          maxLength={6}
          style={{
            width: "100%",
            padding: "15px",
            margin: "15px 0",
            borderRadius: "20px",
            border: "none",
            textAlign: "center",
            fontSize: "24px",
            fontWeight: "bold",
            letterSpacing: "8px",
            color: darkPurple,
            backgroundColor: white,
            boxShadow: `4px 4px 0px ${darkPurple}`,
            fontFamily: clearCustomFont,
            outline: "none"
          }}
          value={authCode}
          onChange={(e) => setAuthCode(e.target.value.replace(/[^0-9]/g, ""))}
          placeholder="000000"
        />

        {/* 메인 버튼: PartialSans */}
        <button 
          style={{
            backgroundColor: white,
            color: darkPurple,
            padding: "12px 30px",
            borderRadius: "20px",
            border: `2px solid ${darkPurple}`,
            cursor: "pointer",
            fontSize: "18px",
            fontWeight: "bold",
            boxShadow: `4px 4px 0px ${darkPurple}`,
            marginTop: "10px",
            width: "100%",
            fontFamily: customFont
          }}
          onClick={handleVerifyCode} 
          disabled={isVerifying || countdown === 0}
        >
          {isVerifying ? "인증 중..." : "인증하고 로그인"}
        </button>

        {/* 링크형 버튼들: SchoolSafetyRoundedSmile */}
        <button
          style={{ 
            backgroundColor: "transparent", border: "none", color: darkPurple, 
            cursor: "pointer", fontSize: "14px", textDecoration: "underline", 
            marginTop: "15px", width: "100%", fontFamily: clearCustomFont 
          }}
          onClick={() => {/* 재발송 로직 */}}
        >
          코드 재발송하기
        </button>

        <button
          style={{ 
            backgroundColor: "transparent", border: "none", color: darkPurple, 
            cursor: "pointer", fontSize: "14px", marginTop: "5px", width: "100%", 
            fontFamily: clearCustomFont 
          }}
          onClick={onRestartSignup}
        >
          가입 다시 시작하기
        </button>

        {/* 메시지: SchoolSafetyRoundedSmile */}
        {message && (
          <div style={{
            marginTop: "15px",
            padding: "10px",
            borderRadius: "10px",
            fontSize: "13px",
            backgroundColor: "#D9534F",
            color: "white",
            fontFamily: clearCustomFont
          }}>
            {message}
          </div>
        )}
      </div>
    </div>
  );
};

export default EmailAuthPage;