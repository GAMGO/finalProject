import React, { useState, useCallback } from "react"; // ⭐️ useCallback 추가
import axios from 'axios';
import dishLogo from "../assets/DISH_LOGO.png"; // ✅ 로고 이미지 import
import { useNavigate } from 'react-router-dom';
import { setAuthToken } from '../api/apiClient';
const baseURL = import.meta.env.VITE_BASE_URL;
// 'onToggleMode' 프롭을 받아 회원가입 버튼 클릭 시 모드를 전환하도록 합니다.
const LoginPage = ({ onToggleMode, onLoginSuccess }) => {
  // ------------------------------------
  // 1. 상태 관리
  // ------------------------------------
  const [customer_id, setcustomer_id] = useState("");
  const [password, setPassword] = useState("");
  const navigate = useNavigate();
  // ------------------------------------
  // 2. 상태 설정 함수 (useCallback)
  // ------------------------------------
  const handleIdChange = useCallback((e) => {
    setcustomer_id(e.target.value);
  }, []);

  const handlePasswordChange = useCallback((e) => {
    setPassword(e.target.value);
  }, []);

  // ------------------------------------
  // 3. 로그인 처리
  // ------------------------------------
  const handleLogin = async () => {

    // ... 기존 로그인 로직 유지 ...
    if (!customer_id || !password) {
      alert("아이디와 비밀번호를 모두 입력해주세요.");
      return;
    }

    // ⭐️ API 호출 데이터 준비
    const loginData = {
      id: customer_id, // ⚠️ 인풋 필드에서 값을 가져오는 변수명인지 확인하세요.
      password: password
    };

    try {
      const response = await axios.post(
        // 백엔드 로그인 엔드포인트: https://api.dishinside.shop/api/auth/login
        `${baseURL}/api/auth/login`,
        loginData,
        { withCredentials: true }
      );

      // ⭐️ 로그인 성공 처리
      alert("로그인 성공!");

      const accessToken = response.data.accessToken || response.data.token;
      if (accessToken) {
        onLoginSuccess(accessToken); 
      } else {
        console.error("로그인 응답에 Access Token이 포함되어 있지 않습니다.");
      }

    } catch (error) {
      // ⭐️ 서버 연결 또는 인증 실패 처리
      if (error.response) {
        // 서버가 응답을 보냈지만 2xx 범위가 아닐 경우 (예: 401 Unauthorized, 400 Bad Request)
        alert(`로그인 실패: ${error.response.data.message || "아이디 또는 비밀번호를 확인해주세요."}`);
        console.error("로그인 에러 응답:", error.response);
      } else if (error.request) {
        // 요청이 전송되었지만 응답을 받지 못한 경우 (네트워크, CORS 등)
        alert("서버 응답이 없습니다. CORS 설정 또는 네트워크 상태를 확인해주세요.");
        console.error("로그인 에러 요청:", error.request);
      } else {
        // 요청 설정 자체에서 오류가 발생한 경우
        alert("서버 연결에 실패했습니다. 네트워크 상태를 확인해주세요.");
        console.error("로그인 에러:", error.message);
      }
    }
  };
  // ------------------------------------
  // 4. 스타일 정의
  // ------------------------------------
  const darkPurple = "#78266A";
  const lightPeach = "#F5D7B7";
  const white = "#FFFFFF";
  const customFont = "PartialSans, sans-serif";

  const fontFaceCss = `
    @font-face {
      font-family: 'PartialSans';
      src: url('https://cdn.jsdelivr.net/gh/projectnoonnu/noonfonts_2307-1@1.1/PartialSansKR-Regular.woff2') format('woff2');
      font-weight: normal;
      font-display: swap;
    }
  `;

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
    marginTop: "20px",
    margin: "5px",
    transition: "background-color 0.3s",
    fontFamily: customFont,
    boxShadow: `4px 4px 0px ${darkPurple}`,
  };

  // ------------------------------------
  // 5. 렌더링
  // ------------------------------------
  return (
    <div style={containerStyle}>
      <style>{fontFaceCss}</style>
      <div style={loginBoxStyle}>
        {/* 로고 영역 */}
        <div>
          {/* ✅ 배포 환경에서도 동작하는 로고 경로 */}
          <img src={dishLogo} alt="DISH 로고" style={logoContainerStyle} />
        </div>

        {/* ID 입력 필드 */}
        <div style={inputGroupStyle}>
          <label htmlFor="customer_ix" style={labelStyle}>
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
