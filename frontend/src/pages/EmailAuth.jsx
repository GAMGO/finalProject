import React, { useState, useEffect, useRef } from "react";
import axios from "axios";

const baseURL = import.meta.env.VITE_BASE_URL;

const EmailAuthPage = ({ registeredEmail, signupPayload, onAuthSuccess, onRestartSignup }) => {
  const [authCode, setAuthCode] = useState("");
  const [countdown, setCountdown] = useState(300);
  const [isVerifying, setIsVerifying] = useState(false);
  const [isResending, setIsResending] = useState(false);
  const [message, setMessage] = useState({ text: "", type: "" });
  const hasSentInitialMail = useRef(false);//중복 발송을 원천 차단하는 불변의 가드
  // 스타일 정의 (사용자 원본 그대로 유지)
  const darkPurple = "#78266A";
  const lightPeach = "#F5D7B7";
  const white = "#FFFFFF";
  const customFont = "PartialSans, SchoolSafetyRoundedSmile, sans-serif";
  const clearCustomFont = "SchoolSafetyRoundedSmile, sans-serif";
  const fontFaceCss = `@font-face { font-family: 'PartialSans'; src: url('https://cdn.jsdelivr.net/gh/projectnoonnu/noonfonts_2307-1@1.1/PartialSansKR-Regular.woff2') format('woff2'); } @font-face { font-family: 'SchoolSafetyRoundedSmile'; src: url('https://cdn.jsdelivr.net/gh/projectnoonnu/2408-5@1.0/HakgyoansimDunggeunmisoTTF-R.woff2') format('woff2'); }`;
  const containerStyle = { display: "flex", justifyContent: "center", alignItems: "center", width: "100vw", height: "100vh", backgroundColor: darkPurple, fontFamily: customFont };
  const boxStyle = { backgroundColor: lightPeach, padding: "60px 40px", borderRadius: "40px", boxShadow: "0 4px 15px rgba(0, 0, 0, 0.3)", width: "45vh", textAlign: "center" };

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

  //최초 발송 중복 요청 물리적 차단 (StrictMode 대응)
  useEffect(() => {
    if (!registeredEmail || hasSentInitialMail.current) return;

    const triggerInitialMail = async () => {
      hasSentInitialMail.current = true; // 통신 시작 전 즉시 잠금
      try {
        const sendUrl = `${baseURL}/api/email/resend?email=${encodeURIComponent(registeredEmail)}`;
        await axios.post(sendUrl, null, { withCredentials: true });
        console.log("[DEBUG] 최초 인증 메일 발송 성공");
      } catch (error) {
        hasSentInitialMail.current = false; // 에러 시에는 다시 보낼 수 있게 해제
        console.error("최초 메일 발송 실패", error);
      }
    };
    triggerInitialMail();
  }, [registeredEmail]);

  const handleResendCode = async () => {
    if (isResending) return;
    setIsResending(true);
    try {
      const resendUrl = `${baseURL}/api/email/resend?email=${encodeURIComponent(registeredEmail)}`;
      await axios.post(resendUrl, null, { withCredentials: true });
      setCountdown(300);
      setMessage({ text: "인증 이메일이 발송되었습니다.", type: "success" });
    } catch (error) {
      setMessage({ text: error.response?.data?.message || "재발송 실패", type: "error" });
    } finally {
      setIsResending(false);
    }
  };

  const handleVerifyCode = async () => {
    if (isVerifying) return;
    setIsVerifying(true);

    try {
      // 1. 이메일 코드 검증
      await axios.post(`${baseURL}/api/email/verify`, {
        email: registeredEmail,
        token: authCode
      }, { withCredentials: true });

      // 2. 가입 처리 (백엔드 수정본 적용: 토큰 수신)
      const res = await axios.post(`${baseURL}/api/auth/signup`, signupPayload, { withCredentials: true });
      // 백엔드 응답에서 Access Token 추출
      const { token } = res.data;
      if (token) {
        // ✅ 인증 성공 상태 확정 - useEffect의 메일 발송 로직 차단
        hasSentInitialMail.current = true;

        // Access Token 저장
        localStorage.setItem("jwtToken", token);

        // ✅ 핵심: AuthCheck.jsx가 즉시 감지하도록 강제 이벤트 발생
        window.dispatchEvent(new Event('storage'));

        setMessage({ text: "인증 성공! 메인 화면으로 이동합니다.", type: "success" });

        // 3. 페이지 전환 지연 실행 (토큰 안착 보장)
        setTimeout(() => {
          // 부모(AuthPage) 상태를 'authSuccess'로 변경하여 App 렌더링 유도
          onAuthSuccess();
          // 강제 리프레시가 필요한 경우: window.location.href = "/";
        }, 500);
      }
    } catch (error) {
      setMessage({ text: error.response?.data?.message || "처리 중 오류 발생", type: "error" });
    } finally {
      setIsVerifying(false);
    }
  };
  // 메일 중복 발송 방지 useEffect 수정
  useEffect(() => {
    // 이미 인증 성공했거나 발송 기록이 있다면 발송 차단
    if (!registeredEmail || hasSentInitialMail.current) return;

    const triggerInitialMail = async () => {
      hasSentInitialMail.current = true; // 통신 직전 잠금
      try {
        await axios.post(`${baseURL}/api/email/resend?email=${encodeURIComponent(registeredEmail)}`, null, { withCredentials: true });
      } catch (error) {
        hasSentInitialMail.current = false; // 실패 시에만 해제
      }
    };
    triggerInitialMail();
  }, [registeredEmail]);

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