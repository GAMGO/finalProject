// src/pages/LoginPage.jsx
import React, { useState, useCallback, useEffect } from "react";
import axios from "axios";
import { useNavigate } from "react-router-dom";
import { setAuthToken } from "../api/apiClient";
import "../theme/theme.css"; // ✅ 테마 CSS

const baseURL = import.meta.env.VITE_BASE_URL;
const dishLogoUrl = "../src/assets/DISH_LOGO.png";

const LoginPage = ({ onToggleMode, onLoginSuccess }) => {
  // ------------------------------------
  // 1. 상태 관리
  // ------------------------------------
  const [customer_id, setcustomer_id] = useState("");
  const [password, setPassword] = useState("");
  const [message, setMessage] = useState({ text: "", type: "" });
  const navigate = useNavigate();

  // ✅ 로그인/회원가입 공통 테마 상태 (light | dark)
  const [theme, setTheme] = useState(() => {
    if (typeof window === "undefined") return "light";
    return localStorage.getItem("theme") || "light";
  });

  // ✅ 테마 변경 시 data-theme + localStorage 반영
  useEffect(() => {
    if (typeof document !== "undefined") {
      document.documentElement.setAttribute("data-theme", theme);
    }
    if (typeof window !== "undefined") {
      localStorage.setItem("theme", theme);
    }
  }, [theme]);

  const toggleTheme = () => {
    setTheme((prev) => (prev === "light" ? "dark" : "light"));
  };

  const isDark = theme === "dark";

  // ------------------------------------
  // 2. 상태 설정 함수 (useCallback)
  // ------------------------------------
  const handleIdChange = useCallback((e) => {
    setcustomer_id(e.target.value);
    setMessage({ text: "", type: "" });
  }, []);

  const handlePasswordChange = useCallback((e) => {
    setPassword(e.target.value);
    setMessage({ text: "", type: "" });
  }, []);

  // ------------------------------------
  // 3. 로그인 처리
  // ------------------------------------
  const handleLogin = async () => {
    if (!customer_id || !password) {
      setMessage({
        text: "아이디와 비밀번호를 모두 입력해주세요.",
        type: "error",
      });
      return;
    }

    const loginData = {
      id: customer_id,
      password: password,
    };

    try {
      const response = await axios.post(
        `${baseURL}/api/auth/login`,
        loginData,
        { withCredentials: true }
      );

      const token = response.data.token;
      const refreshToken = response.data.refreshToken;

      if (token) {
        if (typeof onLoginSuccess === "function") {
          setAuthToken(token, refreshToken);
          setMessage({ text: "로그인 성공!", type: "success" });
          setTimeout(() => {
            onLoginSuccess();
          }, 1500);
        } else {
          console.error(
            "onLoginSuccess props가 유효한 함수가 아닙니다. 라우팅 설정 확인 필요."
          );
        }
      } else {
        setMessage({
          text: "로그인 응답에 Access Token이 포함되어 있지 않습니다.",
          type: "error",
        });
        console.error("로그인 응답에 Access Token이 포함되어 있지 않습니다.");
      }
    } catch (error) {
      let errorMessage =
        "서버 연결에 실패했습니다. 네트워크 상태를 확인해주세요.";

      if (error.response) {
        errorMessage =
          error.response.data.message || "아이디 또는 비밀번호를 확인해주세요.";
        console.error("로그인 에러 응답:", error.response);
      } else if (error.request) {
        errorMessage =
          "서버 응답이 없습니다. (CORS 문제 가능성 높음) 백엔드 서버의 CORS 설정을 확인해주세요.";
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
  const logoDarkBrown = "#5e3a31"; // 로고 카드 색
  const chocolateShadow = "#3a221c"; // 그림자용 짙은 갈색

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
  const fontClearCss = `
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
}`;
  const fontSet = [fontClearCss, fontFaceCss];

  const textShadowStyle = { textShadow: `4px 4px 2px ${darkPurple}` };

  const containerStyle = {
    display: "flex",
    justifyContent: "center",
    alignItems: "center",
    width: "100vw",
    height: "100vh",
    backgroundColor: isDark ? "#4A1010" : darkPurple, // 다크 모드 적갈색
    fontFamily: customFont,
    color: isDark ? "#fef3e8" : "#222222",
  };
  const loginBoxStyle = {
    backgroundColor: isDark ? logoDarkBrown : lightPeach, // 카드색
    padding: "60px 40px",
    borderRadius: "40px",
    boxShadow: isDark
      ? "0 10px 30px rgba(0, 0, 0, 0.7)"
      : "0 4px 15px rgba(0, 0, 0, 0.3)",
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

  // ✅ 다크 모드: 입력칸 연한 갈색, 글자는 진한 갈색
  const inputStyle = {
    width: "100%",
    padding: "12px 10px",
    margin: "8px 0",
    border: "none",
    borderRadius: "20px",
    boxSizing: "border-box",
    outline: "none",
    fontSize: "16px",
    backgroundColor: isDark ? lightPeach : white,
    color: isDark ? logoDarkBrown : darkPurple,
    fontFamily: clearCustomFont,
    fontWeight: 700,
    boxShadow: `4px 4px 0px ${
      isDark ? chocolateShadow : darkPurple
    }`,
  };

  const buttonStyle = {
    backgroundColor: isDark ? lightPeach : white,
    color: isDark ? logoDarkBrown : darkPurple,
    padding: "10px 30px",
    fontSize: "18px",
    fontWeight: "100",
    borderRadius: "20px",
    border: `2px solid ${isDark ? chocolateShadow : darkPurple}`,
    cursor: "pointer",
    marginTop: "20px",
    margin: "5px",
    transition: "background-color 0.3s",
    fontFamily: customFont,
    boxShadow: `4px 4px 0px ${
      isDark ? chocolateShadow : darkPurple
    }`,
  };

  const messageStyle = {
    marginTop: "15px",
    marginBottom: "15px",
    padding: "10px",
    borderRadius: "10px",
    fontWeight: "0",
    color: white,
    fontFamily: clearCustomFont,
    backgroundColor: message.type === "error" ? "#D9534F" : "#5CB85C",
    fontSize: "14px",
  };

  // ------------------------------------
  // 5. 렌더링
  // ------------------------------------
  return (
    <div style={containerStyle}>
      <style>{fontSet}</style>

      {/* ✅ 로그인 화면용 라이트/다크 토글 */}
      <button
        type="button"
        className="side-nav-btn theme-toggle-btn auth-theme-toggle"
        onClick={toggleTheme}
      >
        {theme === "light" ? "다크 모드" : "라이트 모드"}
      </button>

      <div style={loginBoxStyle}>
        <div>
          <img src={dishLogoUrl} alt="DISH 로고" style={logoContainerStyle} />
        </div>

        {/* 메시지 영역 */}
        {message.text && <div style={messageStyle}>{message.text}</div>}

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
