import React, { useState, useCallback } from "react"; // ⭐️ useCallback 추가

// 'onToggleMode' 프롭을 받아 로그인 버튼 클릭 시 모드를 전환하도록 합니다.
const SignupPage = ({ onToggleMode }) => {
  // ------------------------------------
  // 1. 상태 관리
  // ------------------------------------
  const [customerId, setCustomerId] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [email, setEmail] = useState("");

  // ------------------------------------
  // 2. 상태 설정 함수 래핑 (안정성 확보)
  // ------------------------------------
  // ⭐️ 범용 입력 핸들러 함수 (필드별 setter를 호출)
  const createHandleChange = useCallback((setter) => (e) => {
    setter(e.target.value);
  }, []); // 이 함수 자체는 한 번만 생성

  // ------------------------------------
  // 3. 회원가입 처리 함수 (생략 및 유지)
  // ------------------------------------
  const handleRegister = async () => {
    // ... 기존 회원가입 로직 유지 ...
    if (!customerId || !password || !confirmPassword || !email) {
      alert("모든 필드를 입력해주세요.");
      return;
    }
    // ... API 호출 및 에러 처리 ...
    try {
        // ...
    } catch (error) {
        alert("서버 연결에 실패했습니다. 네트워크 상태를 확인해주세요.");
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
    marginTop: "10px", margin: "5px", 
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

        {/* 1. ID 입력 필드 */}
        <div style={inputGroupStyle}>
          <label htmlFor="reg_customerId" style={labelStyle}>ID</label>
          <input
            type="text" id="reg_customerId" placeholder="사용할 아이디를 입력하세요" style={inputStyle}
            value={customerId} 
            onChange={createHandleChange(setCustomerId)} // ⭐️ useCallback 함수 적용
          />
        </div>

        {/* 2. Email 입력 필드 */}
        <div style={inputGroupStyle}>
          <label htmlFor="reg_email" style={labelStyle}>Email</label>
          <input
            type="email" id="reg_email" placeholder="이메일 주소를 입력하세요" style={inputStyle}
            value={email} 
            onChange={createHandleChange(setEmail)} // ⭐️ useCallback 함수 적용
          />
        </div>

        {/* 3. 비밀번호 입력 필드 */}
        <div style={inputGroupStyle}>
          <label htmlFor="reg_password" style={labelStyle}>비밀번호</label>
          <input
            type="password" id="reg_password" placeholder="비밀번호를 입력하세요" style={inputStyle}
            value={password} 
            onChange={createHandleChange(setPassword)} // ⭐️ useCallback 함수 적용
          />
        </div>
        
        {/* 4. 비밀번호 확인 입력 필드 */}
        <div style={inputGroupStyle}>
          <label htmlFor="reg_confirmPassword" style={labelStyle}>비밀번호 확인</label>
          <input
            type="password" id="reg_confirmPassword" placeholder="비밀번호를 다시 입력하세요" style={inputStyle}
            value={confirmPassword} 
            onChange={createHandleChange(setConfirmPassword)} // ⭐️ useCallback 함수 적용
          />
        </div>

        {/* 버튼 영역 */}
        <div>
          <button type="button" style={buttonStyle} onClick={handleRegister}>
            회원가입 완료
          </button>
          
          <button type="button" style={buttonStyle} onClick={onToggleMode}>
            로그인 페이지로
          </button>
        </div>
      </div>
    </div>
  );
};

export default SignupPage;