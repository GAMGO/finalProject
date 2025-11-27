import React, { useState, useCallback } from "react";
import axios from "axios";
import dishLogo from "../assets/DISH_LOGO.png"; // ✅ 로고 이미지 import

// const API_BASE_URL = process.env.REACT_APP_API_BASE_URL;
const API_BASE_URL = "https://api.dishinside.shop";

const LoginPage = ({ onToggleMode }) => {
  // ------------------------------------
  // 1. 상태 관리
  // ------------------------------------
  const [customerId, setCustomerId] = useState("");
  const [password, setPassword] = useState("");

  // ------------------------------------
  // 2. 상태 설정 함수 (useCallback)
  // ------------------------------------
  const handleIdChange = useCallback((e) => {
    setCustomerId(e.target.value);
  }, []);

  const handlePasswordChange = useCallback((e) => {
    setPassword(e.target.value);
  }, []);

  // ------------------------------------
  // 3. 로그인 처리
  // ------------------------------------
  const handleLogin = async () => {
    if (!customerId || !password) {
      alert("아이디와 비밀번호를 모두 입력해주세요.");
      return;
    }

    const loginData = {
      id: customerId,        // 👈 백엔드에서 기대하는 필드명 확인
      password_hash: password,
    };

    try {
      const response = await axios.post(
        `${API_BASE_URL}/api/auth/login`,
        loginData,
        { withCredentials: true }
      );

      alert("로그인 성공!");
      console.log("로그인 응답 데이터:", response.data);

      // TODO: JWT 저장 및 라우팅 등 성공 후 처리
      // localStorage.setItem("jwtToken", response.data.token);
      // navigate("/");

    } catch (error) {
      if (error.response) {
        alert(
          `로그인 실패: ${
            error.response.data.message || "아이디 또는 비밀번호를 확인해주세요."
          }`
        );
        console.error("로그인 에러 응답:", error.response);
      } else if (error.request) {
        alert("서버 응답이 없습니다. CORS 설정 또는 네트워크 상태를 확인해주세요.");
        console.error("로그인 에러 요청:", error.request);
      } else {
        alert("서버 연결에 실패했습니다. 네트워크 상태를 확인해주세요.");
        console.error("로그인 에러:", error.message);
      }
    }
  };

  // ------------------------------------
  // 4. 스타일 정의
  // ------------------------------------
  const darkPurple = "#5B2C6F";
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
    fontWeight: "bold",
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
          <label htmlFor="customerId" style={labelStyle}>
            ID
          </label>
          <input
            type="text"
            id="customerId"
            placeholder="아이디를 입력하세요"
            style={inputStyle}
            value={customerId}
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
