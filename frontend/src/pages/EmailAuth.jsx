import React, { useState, useEffect, useCallback } from "react";
import axios from "axios";

// API 기본 URL 설정 (SignupPage와 동일해야 합니다)
const baseURL = import.meta.env.VITE_BASE_URL;

// ----------------------------------------------------------------------
// 1. 공통 스타일 정의 (AuthPage의 스타일과 일치하도록)
// ----------------------------------------------------------------------
const darkPurple = "#78266A";
const deepDarkPurple = "#5B2C6F";
const white = "#FFFFFF";
const customFont = "PartialSans, sans-serif";

const textShadowStyle = { textShadow: `4px 4px 2px ${darkPurple}` };

const titleStyle = {
  fontSize: "32px",
  fontWeight: "100",
  color: white,
  margin: "25px",
  display: "block",
  letterSpacing: "2px",
  ...textShadowStyle,
};

const inputStyle = {
  width: "100%",
  padding: "12px 10px",
  margin: "8px 0",
  border: "none",
  borderRadius: "20px",
  boxSizing: "border-box",
  outline: "none",
  fontSize: "16px",
  backgroundColor: white,
  color: darkPurple,
  fontFamily: customFont,
  boxShadow: `4px 4px 0px ${darkPurple}`,
};

const buttonStyle = {
  backgroundColor: white,
  color: darkPurple,
  padding: "10px 30px",
  fontSize: "18px",
  fontWeight: "100",
  borderRadius: "20px",
  border: `2px solid ${darkPurple}`,
  cursor: "pointer",
  marginTop: "10px",
  margin: "5px",
  transition: "background-color 0.3s",
  fontFamily: customFont,
  boxShadow: `4px 4px 0px ${darkPurple}`,
};

const secondaryButtonStyle = {
  ...buttonStyle,
  backgroundColor: 'transparent',
  border: 'none',
  color: deepDarkPurple,
  boxShadow: 'none',
  padding: '8px 0',
  fontSize: '15px',
};


const inputGroupStyle = { marginBottom: "20px", textAlign: "left" };

// ----------------------------------------------------------------------
// 2. 유틸리티 함수
// ----------------------------------------------------------------------

// 시간 포맷팅 함수 (MM:SS 형식)
const formatTime = (seconds) => {
  const minutes = Math.floor(seconds / 60);
  const remainingSeconds = seconds % 60;
  return `${minutes.toString().padStart(2, '0')}:${remainingSeconds.toString().padStart(2, '0')}`;
};

// 범용 입력 핸들러 함수
const createHandleChange = (setter) => (e) => setter(e.target.value);


// ----------------------------------------------------------------------
// 3. EmailAuthPage 컴포넌트
// ----------------------------------------------------------------------

const EmailAuthPage = ({ registeredEmail, onAuthSuccess, onRestartSignup }) => {
  const [authCode, setAuthCode] = useState('');
  const [countdown, setCountdown] = useState(300); // 5분 = 300초
  const [isVerifying, setIsVerifying] = useState(false);
  const [isResending, setIsResending] = useState(false);
  const [message, setMessage] = useState(''); // 사용자 메시지 표시용

  // 카운트다운 타이머 설정
  useEffect(() => {
    // 0초가 되면 경고를 띄우고 재시작 요청
    if (countdown <= 0) {
        if (countdown === 0) {
            setMessage("인증 시간이 만료되었습니다. 재발송하거나 다시 회원가입해 주세요.");
        }
        return;
    }

    // 1초마다 카운트다운
    const interval = setInterval(() => {
      setCountdown(prev => prev - 1);
    }, 1000);

    // 컴포넌트 unmount 또는 countdown 변경 시 타이머 정리
    return () => clearInterval(interval);
  }, [countdown, onRestartSignup]);


  // 인증 코드 검증 핸들러
  const handleVerifyCode = async () => {
    setMessage('');
    if (!authCode || authCode.length !== 6) {
      setMessage("6자리 인증번호를 정확히 입력해주세요.");
      return;
    }
    if (countdown === 0) {
        setMessage("인증 시간이 만료되었습니다. 인증을 다시 시도해주세요.");
        return;
    }

    setIsVerifying(true);
    try {
      // ⭐️ 백엔드 DTO에 맞게 'code' 대신 'token' 키 사용 (수정 1)
      const verifyData = {
        email: registeredEmail,
        token: authCode, 
      };

      // ⭐️ API 경로를 /api/email/verify로 수정 (수정 2)
      await axios.post(
        `${baseURL}/api/email/verify`, 
        verifyData,
        { withCredentials: true }
      );

      // 🚨 alert() 사용 대신 커스텀 메시지 사용 (추가 수정)
      setMessage("인증에 성공했습니다! 자동으로 로그인됩니다.");
      setTimeout(() => onAuthSuccess(), 1000); // 메시지를 보여준 후 1초 뒤 전환

    } catch (error) {
      if (error.response && error.response.status === 400) {
        // 백엔드 응답 메시지 사용
        setMessage(error.response.data.message || "인증번호가 유효하지 않거나 만료되었습니다.");
      } else {
        setMessage("인증 서버에 문제가 발생했습니다. 다시 시도해 주세요.");
      }
      console.error("인증 에러:", error.response || error.message);
    } finally {
      setIsVerifying(false);
    }
  };
  
  // 인증 코드 재발송 핸들러
  const handleResendCode = async () => {
    setMessage('');
    setIsResending(true);
    try {
      // ⭐️ API 경로 /api/email/resend 호출 (POST, @RequestParam)
      await axios.post(
        `${baseURL}/api/email/resend?email=${registeredEmail}`, 
        null, // @RequestParam 이므로 body는 null
        { withCredentials: true }
      );

      // 성공 시 타이머를 5분으로 재설정
      setCountdown(300); 
      setMessage("새로운 인증 이메일이 발송되었습니다. 5분 안에 인증해 주세요.");

    } catch (error) {
      if (error.response && error.response.status === 400) {
        setMessage(error.response.data.message || "재발송 요청에 실패했습니다.");
      } else {
        setMessage("재발송 처리 중 서버 오류가 발생했습니다.");
      }
      console.error("재발송 에러:", error.response || error.message);
    } finally {
      setIsResending(false);
    }
  };


  return (
    <>
      <h2 style={{...titleStyle, marginBottom: '40px'}}>이메일 인증</h2>
      <p style={{ color: deepDarkPurple, marginBottom: '20px', fontSize: '16px' }}>
        회원님 이메일 주소
        <span style={{ fontWeight: 'bold', fontSize: '18px', display: 'block', marginTop: '5px' }}>
          {registeredEmail}
        </span>
        (으)로 6자리 인증 코드를 전송했습니다.
      </p>

      {/* 타이머 표시 영역 */}
      <div style={{ marginBottom: '30px', backgroundColor: darkPurple, padding: '15px', borderRadius: '15px' }}>
        <p style={{ 
            fontSize: '56px', 
            fontWeight: 'bold', 
            color: countdown <= 20 ? '#FF5555' : white, 
        }}>
          {formatTime(countdown)}
        </p>
        <p style={{ color: white, fontSize: '14px', marginTop: '5px' }}>
          남은 인증 유효 시간
        </p>
      </div>

      {/* 인증 코드 입력 필드 */}
      <div style={inputGroupStyle}>
        <input
          type="text"
          maxLength="6"
          placeholder="6자리 코드를 입력하세요"
          style={{
              ...inputStyle,
              textAlign: 'center',
              fontSize: '24px',
              letterSpacing: '0.5em', // 코드 구분을 위한 자간
              padding: '15px 10px',
              borderRadius: '10px'
          }}
          value={authCode}
          onChange={createHandleChange(setAuthCode)}
          disabled={countdown === 0 || isVerifying || isResending}
        />
      </div>

      {/* 메시지 영역 */}
      {message && (
        <p style={{ color: countdown <= 20 ? '#FF5555' : deepDarkPurple, textAlign: 'center', marginBottom: '15px', fontWeight: 'bold' }}>
          {message}
        </p>
      )}

      {/* 재발송 버튼 */}
      <button 
        type="button" 
        onClick={handleResendCode}
        style={{
          ...secondaryButtonStyle,
          width: '100%',
          cursor: isVerifying || isResending ? 'not-allowed' : 'pointer',
          color: isVerifying || isResending ? '#aaa' : deepDarkPurple,
        }}
        disabled={isVerifying || isResending}
      >
        {isResending ? '재발송 중...' : '인증 이메일 재발송'}
      </button>

      {/* 인증 버튼 */}
      <div>
        <button 
          type="button" 
          onClick={handleVerifyCode}
          style={{
            ...buttonStyle,
            width: '100%',
            padding: '15px 0',
            fontSize: '20px',
            backgroundColor: countdown === 0 || isVerifying || isResending ? '#ccc' : white,
            color: countdown === 0 || isVerifying || isResending ? '#666' : darkPurple,
            cursor: countdown === 0 || isVerifying || isResending ? 'not-allowed' : 'pointer',
            boxShadow: countdown === 0 || isVerifying || isResending ? 'none' : `4px 4px 0px ${darkPurple}`,
            border: countdown === 0 || isVerifying || isResending ? 'none' : `2px solid ${darkPurple}`,
          }}
          disabled={countdown === 0 || isVerifying || isResending}
        >
          {isVerifying ? '인증 중...' : '인증하고 자동 로그인'}
        </button>
        
        <button 
          type="button" 
          onClick={onRestartSignup}
          style={{...secondaryButtonStyle, width: '100%', marginTop: '15px', color: deepDarkPurple}}
          disabled={isVerifying || isResending}
        >
          회원가입 다시 시작
        </button>
      </div>
    </>
  );
};

export default EmailAuthPage;