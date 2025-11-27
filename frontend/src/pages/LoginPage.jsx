import React, { useState, useCallback } from "react"; // ⭐️ useCallback 추가
import axios from 'axios';
//const API_BASE_URL = process.env.REACT_APP_API_BASE_URL;
const API_BASE_URL = "https://api.dishinside.shop";
// 'onToggleMode' 프롭을 받아 회원가입 버튼 클릭 시 모드를 전환하도록 합니다.
const LoginPage = ({ onToggleMode }) => { 
  // ------------------------------------
  // 1. 상태 관리
  // ------------------------------------
  const [customerId, setCustomerId] = useState("");
  const [password, setPassword] = useState("");

  // ------------------------------------
  // 2. 상태 설정 함수 래핑 (안정성 확보)
  // ------------------------------------
  // ⭐️ ID 입력 핸들러를 useCallback으로 메모이제이션
  const handleIdChange = useCallback((e) => {
      setCustomerId(e.target.value);
  }, []);

  // ⭐️ PW 입력 핸들러를 useCallback으로 메모이제이션
  const handlePasswordChange = useCallback((e) => {
      setPassword(e.target.value);
  }, []);
  
  const handleLogin = async () => {
  // ... 기존 로그인 로직 유지 ...
  if (!customerId || !password) {
    alert("아이디와 비밀번호를 모두 입력해주세요.");
    return;
  }
  
  // ⭐️ API 호출 데이터 준비
  const loginData = {
    customer_id: customerId, // ⚠️ 인풋 필드에서 값을 가져오는 변수명인지 확인하세요.
    password_hash: password      
  };

  try {
    const response = await axios.post(
      // ⭐️ 백엔드 로그인 엔드포인트: https://api.dishinside.shop/api/auth/login
      `${API_BASE_URL}/api/auth/login`, 
      loginData,
      // ⭐️ 인증 정보를 포함하여 요청합니다.
      { withCredentials: true } 
    );
    
    // ⭐️ 로그인 성공 처리
    alert("로그인 성공!");
    console.log("로그인 응답 데이터:", response.data);
    
    // ⭐️ [성공 후 로직] JWT 토큰 저장 및 페이지 이동 등을 여기에 추가하세요.
    // 예: localStorage.setItem('jwtToken', response.data.token);
    // 예: navigate('/'); 

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
  // 4. 스타일 정의 (기존 인라인 스타일 유지)
  // ------------------------------------
  // ... (fontFaceCss, containerStyle 등 모든 스타일 정의 코드는 그대로 유지) ...
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
    display: "flex", justifyContent: "center", alignItems: "center",
    width: "100vw", height: "100vh", backgroundColor: darkPurple, fontFamily: customFont,
  };
  const loginBoxStyle = {
    backgroundColor: lightPeach, padding: "60px 40px", borderRadius: "40px",
    boxShadow: "0 4px 15px rgba(0, 0, 0, 0.3)", width: "45vh", textAlign: "center", fontFamily: customFont,
  };
  const logoContainerStyle = { maxWidth: "100%", height: "auto" };
  const inputGroupStyle = { marginBottom: "20px", textAlign: "left" };
  const labelStyle = {
    fontSize: "18px", fontWeight: "bold", color: white, marginBottom: "5px",
    display: "block", letterSpacing: "2px", ...textShadowStyle,
  };
  const inputStyle = {
    width: "100%", padding: "12px 10px", margin: "8px 0", border: "none",
    borderRadius: "20px", boxSizing: "border-box", outline: "none", fontSize: "16px",
    backgroundColor: white, color: darkPurple, fontFamily: customFont,
    boxShadow: `4px 4px 0px ${darkPurple}`,
  };
  const buttonStyle = {
    backgroundColor: white, color: darkPurple, padding: "10px 30px",
    fontSize: "18px", fontWeight: "bold", borderRadius: "20px",
    border: `2px solid ${darkPurple}`, cursor: "pointer", 
    marginTop: "20px", margin: "5px", 
    transition: "background-color 0.3s", fontFamily: customFont,
    boxShadow: `4px 4px 0px ${darkPurple}`,
  };
  
  // ------------------------------------
  // 5. 컴포넌트 렌더링 (적용)
  // ------------------------------------
  return (
    <div style={containerStyle}>
      <style>{fontFaceCss}</style>
      <div style={loginBoxStyle}>
        {/* 로고 영역 */}
        <div>
          <img src="..\src\assets\DISH_LOGO.png" alt="DISH 로고" style={logoContainerStyle} />
        </div>

        {/* ID 입력 필드 */}
        <div style={inputGroupStyle}>
          <label htmlFor="customerId" style={labelStyle}>ID</label>
          <input
            type="text" id="customerId" placeholder="아이디를 입력하세요" style={inputStyle}
            value={customerId} 
            onChange={handleIdChange} // ⭐️ useCallback 함수 적용
          />
        </div>

        {/* PW 입력 필드 */}
        <div style={inputGroupStyle}>
          <label htmlFor="password" style={labelStyle}>PW</label>
          <input
            type="password" id="password" placeholder="비밀번호를 입력하세요" style={inputStyle}
            value={password} 
            onChange={handlePasswordChange} // ⭐️ useCallback 함수 적용
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