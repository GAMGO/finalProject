import React, { useState, useCallback } from "react";
import axios from 'axios';
import { useNavigate } from 'react-router-dom';
const baseURL = import.meta.env.VITE_LOCAL_BASE_URL;
const dishLogoUrl = "/src/assets/DISH_LOGO.png";

const LoginPage = ({ onToggleMode, onLoginSuccess }) => {
  // ------------------------------------
  // 1. 상태 관리
  // ------------------------------------
  const [customer_id, setcustomer_id] = useState("");
  const [password, setPassword] = useState("");
  const [message, setMessage] = useState({ text: "", type: "" }); // { text: 메시지 내용, type: 'success' | 'error' }
  const navigate = useNavigate();

  // ------------------------------------
  // 2. 상태 설정 함수 (useCallback)
  // ------------------------------------
  const handleIdChange = useCallback((e) => {
    setcustomer_id(e.target.value);
    setMessage({ text: "", type: "" }); // 입력 시 메시지 초기화
  }, []);

  const handlePasswordChange = useCallback((e) => {
    setPassword(e.target.value);
    setMessage({ text: "", type: "" }); // 입력 시 메시지 초기화
  }, []);

  // ------------------------------------
  // 3. 로그인 처리
  // ------------------------------------
  const handleLogin = async () => {
    if (!customer_id || !password) {
      setMessage({ text: "아이디와 비밀번호를 모두 입력해주세요.", type: "error" });
      return;
    }

    const loginData = {
      id: customer_id,
      password: password
    };

    try {
      // 2. ⭐️ API 호출
      const response = await axios.post(
        `${baseURL}/api/auth/login`,
        loginData,
        { withCredentials: true }
      );

      // 3. ⭐️ 로그인 성공 처리
      const accessToken = response.data.token;
      if (accessToken) {
        // 🚨 onLoginSuccess 함수 유효성 체크
        if (typeof onLoginSuccess === 'function') {
          onLoginSuccess(accessToken); // 유효한 함수일 때만 호출
        } else {
          // 라우팅 충돌로 인한 props 누락 경고
          console.error("onLoginSuccess props가 유효한 함수가 아닙니다. 라우팅 설정 확인 필요.");
        }
        setMessage({ text: "로그인 성공!", type: "success" });
      } else {
        setMessage({ text: "로그인 응답에 Access Token이 포함되어 있지 않습니다.", type: "error" });
        console.error("로그인 응답에 Access Token이 포함되어 있지 않습니다.");
      }
    } catch (error) {
      // 4. ⭐️ 서버 연결 또는 인증 실패 처리
      let errorMessage = "서버 연결에 실패했습니다. 네트워크 상태를 확인해주세요.";

      if (error.response) {
        errorMessage = error.response.data.message || "아이디 또는 비밀번호를 확인해주세요.";
        console.error("로그인 에러 응답:", error.response);
      } else if (error.request) {
        errorMessage = "서버 응답이 없습니다. (CORS 문제 가능성 높음) 백엔드 서버의 CORS 설정을 확인해주세요.";
        console.error("로그인 에러 요청 (CORS/네트워크):", error.request);
      } else {
        errorMessage = `요청 오류: ${error.message}`;
        console.error("로그인 에러:", error.message);
      }

      setMessage({ text: `로그인 실패: ${errorMessage}`, type: "error" });
    }
  };
  // ------------------------------------
  // 4. 스타일 정의
  // ------------------------------------
  const darkPurple = "#78266A";
  const lightPeach = "#F5D7B7";
  const white = "#FFFFFF";
  const customFont = "PartialSans,SchoolSafetyRoundedSmile,sans-serif";
  const clearCustomFont = "SchoolSafetyRoundedSmile,sans-serif";

  const fontFaceCss = `
    @font-face {
      font-family: 'PartialSans';
      src: url('https://cdn.jsdelivr.net/gh/projectnoonnu/noonfonts_2307-1@1.1/PartialSansKR-Regular.woff2') format('woff2');
      font-weight: normal;
      font-display: swap;
    }
  `;
  const fontClearCss = `@font-face {
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
}`;
  const fontSet = [fontClearCss, fontFaceCss];
  const textShadowStyle = { textShadow: `4px 4px 2px ${darkPurple}` };
  const containerStyle = {
    display: "flex",
    justifyContent: "center",
    alignItems: "center",
    width: "100vw",
    height: "100vh",
    backgroundColor: darkPurple,
    fontFamily: customFont,
  };

  const loginBoxStyle = {
    backgroundColor: lightPeach,
    padding: "60px 40px",
    borderRadius: "40px",
    boxShadow: "0 4px 15px rgba(0, 0, 0, 0.3)",
    width: "45vh",
    textAlign: "center",
    fontFamily: customFont,
  };

  const logoContainerStyle = {
    maxWidth: "100%",
    height: "auto",
    marginBottom: "30px",
  };

  const inputGroupStyle = {
    marginBottom: "20px",
    textAlign: "left",
  };

  const labelStyle = {
    fontSize: "18px",
    fontWeight: "bold",
    color: white,
    marginBottom: "5px",
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
    fontFamily: clearCustomFont,
    fontWeight: 700,
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
    marginTop: "20px",
    margin: "5px",
    transition: "background-color 0.3s",
    fontFamily: customFont,
    boxShadow: `4px 4px 0px ${darkPurple}`,
  };

  const messageStyle = {
    marginTop: '15px',
    marginBottom: '15px',
    padding: '10px',
    borderRadius: '10px',
    fontWeight: '0',
    color: white,
    fontFamily: clearCustomFont,
    backgroundColor: message.type === 'error' ? '#D9534F' : '#5CB85C', // 빨간색 또는 초록색
    fontSize: '14px'
  };

  // ------------------------------------
  // 5. 렌더링
  // ------------------------------------
  return (
    <div style={containerStyle}>
      <style>{fontSet}</style>
      <div style={loginBoxStyle}>
        <div>
          <img src={dishLogoUrl} alt="DISH 로고" style={logoContainerStyle} />
        </div>

        {/* 메시지 영역 */}
        {message.text && (
          <div style={messageStyle}>
            {message.text}
          </div>
        )}

        {/* ID 입력 필드 */}
        <div style={inputGroupStyle}>
          <label htmlFor="customer_id" style={labelStyle}>
            ID
          </label>
          <input
            type="text"
            id="customer_id"
            placeholder="아이디를 입력하세요"
            style={inputStyle}
            value={customer_id}
            onChange={handleIdChange}
          />
        </div>

        {/* PW 입력 필드 */}
        <div style={inputGroupStyle}>
          <label htmlFor="password" style={labelStyle}>
            PW
          </label>
          <input
            type="password"
            id="password"
            placeholder="비밀번호를 입력하세요"
            style={inputStyle}
            value={password}
            onChange={handlePasswordChange}
          />
        </div>

        {/* 버튼 영역 */}
        <div>
          <button type="button" style={buttonStyle} onClick={handleLogin}>
            로그인
          </button>

          <button type="button" style={buttonStyle} onClick={onToggleMode}>
            회원가입
          </button>
        </div>
      </div>
    </div>
  );
};

export default LoginPage;