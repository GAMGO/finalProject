import React, { useState, useEffect } from "react";

import axios from "axios";

const baseURL = import.meta.env.VITE_LOCAL_BASE_URL;

const RecoveringPage = ({ onAuthSuccess }) => {
  // URL 쿼리 파라미터에서 토큰을 추출하여 초기값으로 설정
  const initialToken = new URLSearchParams(window.location.search).get("token") || "";
  const [token, setToken] = useState(initialToken);
  const [message, setMessage] = useState({ text: "", type: "" });

  const handleRestore = async () => {
    // 토큰이 없거나 6자리가 아니면 복구 시도를 막음 (선택적)
    if (!token || token.length !== 6) {
      setMessage({ text: "유효하지 않은 코드 형식입니다.", type: "error" });
      return;
    }

    try {
      await axios.post(`${baseURL}/api/auth/restore`, { recoveryToken: token });
      setMessage({ text: "복구 성공! 로그인 페이지로 이동합니다.", type: "success" });
      if (typeof onAuthSuccess === 'function') {
        setTimeout(() => onAuthSuccess(), 2000); // 2초 후 onAuthSuccess() 호출
      } else {
        console.error("onAuthSuccess prop이 함수가 아닙니다. 리다이렉션을 건너뜁니다.");
      }

    } catch (error) {
      // 서버에서 400 Bad Request 등을 반환할 때
      // 이 부분은 서버 오류(400) 처리 시에도 메시지를 출력합니다.
      setMessage({ text: "유효하지 않은 코드입니다.", type: "error" });
    }
  };
  // (중략: 테마 로직)
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

  // URL 파라미터에 토큰이 있을 경우, 자동으로 복구 시도 (선택적)
  // 유저가 직접 인증번호를 입력하도록 유도하기 위해 이 자동 실행 로직을 주석 처리합니다.
  useEffect(() => {
    if (initialToken && initialToken.length === 6) {
      handleRestore();
    }
  }, []);


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
  const containerStyle = {
    display: "flex",
    justifyContent: "center", // 가로 중앙 정렬
    alignItems: "center", // 세로 중앙 정렬
    width: "100vw", // 전체 뷰포트 너비
    height: "100vh", // 전체 뷰포트 높이
    backgroundColor: isDark ? "#4A1010" : darkPurple, // 다크 모드 적갈색
    fontFamily: customFont,
    color: isDark ? "#fef3e8" : "#222222",
  };
  const boxStyle = { backgroundColor: "#F5D7B7", padding: "60px 40px", borderRadius: "40px", width: "45vh", textAlign: "center" };
  const inputStyle = { width: "100%", padding: "12px", borderRadius: "20px", border: "none", boxShadow: `4px 4px 0px #78266A`, textAlign: "center", fontSize: "20px", letterSpacing: "5px" };
  const buttonStyle = { backgroundColor: "#FFFFFF", color: "#78266A", padding: "10px 30px", borderRadius: "20px", border: "2px solid #78266A", marginTop: "20px", cursor: "pointer", boxShadow: "4px 4px 0px #78266A" };
  const fontSet = [fontClearCss, fontFaceCss];
  return (
    <div style={containerStyle}>
      <style>{fontSet}</style>
      <div style={boxStyle}>
        <h2 style={{ color: "#78266A" }}>계정 복구</h2>
        <p style={{ color: "#78266A", fontSize: "14px", marginBottom: "20px" }}>메일로 발송된 6자리 복구 코드를 입력해주세요.</p>
        <input
          style={inputStyle}
          maxLength={6}
          value={token}
          onChange={(e) => setToken(e.target.value)}
        />
        <button style={buttonStyle} onClick={handleRestore}>복구하기</button>
        {message.text && <div style={{ marginTop: "15px", padding: "10px", backgroundColor: message.type === "error" ? "#D9534F" : "#5CB85C", color: "white", borderRadius: "10px" }}>{message.text}</div>}
      </div></div>
  );
};

export default RecoveringPage;