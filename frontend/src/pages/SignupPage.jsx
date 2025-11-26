import React, { useState } from "react";

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
  // 2. 회원가입 처리 함수
  // ------------------------------------
  const handleRegister = async () => {
    if (!customerId || !password || !confirmPassword || !email) {
      alert("모든 필드를 입력해주세요.");
      return;
    }

    if (password !== confirmPassword) {
      alert("비밀번호가 일치하지 않습니다.");
      return;
    }

    try {
      const response = await fetch("/api/auth/register", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ id: customerId, password: password, email: email }),
      });

      if (response.ok) {
        alert("회원가입에 성공했습니다! 로그인 페이지로 돌아갑니다.");
        // 성공 후 로그인 모드로 자동 전환 요청
        onToggleMode(); 
      } else {
        const errorData = await response.json();
        alert(
          `회원가입 실패: ${
            errorData.message || "이미 존재하는 아이디이거나 서버 오류입니다."
          }`
        );
      }
    } catch (error) {
      alert("서버 연결에 실패했습니다. 네트워크 상태를 확인해주세요.");
    }
  };
  
  // ------------------------------------
  // 3. 스타일 정의 (LoginPage와 동일한 스타일 사용 - 복원)
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
  // 4. 컴포넌트 렌더링
  // ------------------------------------
  return (
    <div style={containerStyle}>
      <style>{fontFaceCss}</style>
      <div style={loginBoxStyle}>
        {/* 1. ID 입력 필드 */}
        <div style={inputGroupStyle}>
          <label htmlFor="reg_customerId" style={labelStyle}>ID</label>
          <input
            type="text" id="reg_customerId" placeholder="사용할 아이디를 입력하세요" style={inputStyle}
            value={customerId} onChange={(e) => setCustomerId(e.target.value)}
          />
        </div>

        {/* 2. Email 입력 필드 */}
        <div style={inputGroupStyle}>
          <label htmlFor="reg_email" style={labelStyle}>Email</label>
          <input
            type="email" id="reg_email" placeholder="이메일 주소를 입력하세요" style={inputStyle}
            value={email} onChange={(e) => setEmail(e.target.value)}
          />
        </div>

        {/* 3. 비밀번호 입력 필드 */}
        <div style={inputGroupStyle}>
          <label htmlFor="reg_password" style={labelStyle}>비밀번호</label>
          <input
            type="password" id="reg_password" placeholder="비밀번호를 입력하세요" style={inputStyle}
            value={password} onChange={(e) => setPassword(e.target.value)}
          />
        </div>
        
        {/* 4. 비밀번호 확인 입력 필드 */}
        <div style={inputGroupStyle}>
          <label htmlFor="reg_confirmPassword" style={labelStyle}>비밀번호 확인</label>
          <input
            type="password" id="reg_confirmPassword" placeholder="비밀번호를 다시 입력하세요" style={inputStyle}
            value={confirmPassword} onChange={(e) => setConfirmPassword(e.target.value)}
          />
        </div>

        {/* 버튼 영역 */}
        <div>
          <button type="button" style={buttonStyle} onClick={handleRegister}>
            회원가입 완료
          </button>
          
          {/* 로그인 페이지로 버튼 클릭 시 부모에게 모드 전환 요청 */}
          <button type="button" style={buttonStyle} onClick={onToggleMode}>
            로그인 페이지로
          </button>
        </div>
      </div>
    </div>
  );
};

export default SignupPage;